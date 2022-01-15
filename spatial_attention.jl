module SpatialAttention
    import Flux
    using CUDA
    #using CUDA.CUFFT: rfft, plan_rfft
    import FFTW
    import Main.WindowFunctions

    export Singlehead, Multihead

    const PorT = Union{Pair{Int, Int}, Tuple{Int, Int}}

    struct Singlehead{T, N}
            # trainable weights
        Wkey::AbstractArray{T, N}
        Wval::AbstractArray{T, N}
        Wout::AbstractArray{T, N}
            # activations
        σ_k
        σ
            # dimensional parameters
        spatial_sizes
        padded_sizes
        radius
            # precomputed stuff for fft-convolution
        window_spectrum    # rfft of window function
        rfft_plan          # linear fft operator (FFTW.plan_rfft or CUDA.CUFFT.plan_rfft)
    end

    # TODO: Multihead attention implementation
    # a simple Flux.Parallel(+, single_heads...)-like layer could work...
    # well, actually not, because of args broadcasting to parallel heads, but
    # anyway it is not a huge problem. The actual problems in that case are
    # unnecessary copies of window_spectrum and rfft_plan. These parameters
    # need to be shared by heads.
    struct Multihead{T, N, M}
        Wkey::NTuple{M, AbstractArray{T, N}}
        Wval::NTuple{M, AbstractArray{T, N}}
        Wout::NTuple{M, AbstractArray{T, N}}
        σ_k
        σ
        spatial_sizes
        padded_sizes
        radius
        window_spectrum
        rfft_plan
    end

    function Singlehead(spatial_sizes, batch_size, sz_wkey,
        sz_wval, n_wout, window; σ_k=identity, σ=identity)
        # unfortunately we need to know beforehand spatial and batch sizes
        # to get an fft plan for v*k'. This makes input args list even messier.
        wargs = map((sz_wkey, sz_wval, last(sz_wval) => n_wout)) do chan
            Flux.convfilter(ntuple(i->1, length(spatial_sizes)), chan)
        end
        radius = WindowFunctions.get_radius(window)
        # nextprod is for faster fft
        padded_sizes = map(d->nextprod([2,3,5], d), spatial_sizes .+ radius)
        kernel_padded = _pad_to_size(WindowFunctions.construct_kernel(window),
                        padded_sizes)
        window_spectrum = FFTW.rfft(kernel_padded)
        # constructing fft plan for v*k' for fast convolution with window.
        # It is possible to just use NNlib.conv for the purpose of convolving
        # with the predefined kernel and not mess around with paddings and ffts
        # but I think that would be slower for large window radii.
        # I'll check that later (<- TODO)
        tmp_vkT = randn(eltype(kernel_padded), padded_sizes..., last(sz_wval),
                    last(sz_wkey), batch_size)
        rfft_plan = FFTW.plan_rfft(tmp_vkT, 1:length(padded_sizes))
        return Singlehead(wargs..., σ_k, σ, spatial_sizes, padded_sizes, radius,
                window_spectrum, rfft_plan)
    end

    Flux.@functor Singlehead (Wkey, Wval, Wout)

    # forward
    function (self::Singlehead)(key, val)
        k1 = self.σ_k.(Flux.conv(key, self.Wkey))
        v1 = Flux.conv(val, self.Wval)
        o1 = spa(k1, v1, self.window_spectrum, self.rfft_plan, self.spatial_sizes,
            self.padded_sizes, self.radius)
        out = self.σ.(Flux.conv(o1, self.Wout))
    end

    function Base.show(io::IO, self::Singlehead{T, N}) where {T, N}
        println(io, "SpatialAttention.Singlehead{$T, $(N-2)D}[
                    radius=$(self.radius),
                    spatial_sizes=$(self.spatial_sizes)...]")
    end

    function spa(k::AbstractArray{T, N}, v::AbstractArray{T,N},
        window_spectrum, rfft_plan, spatial_dims, padded_dims, radius) where {T, N}
        c1, c2, b = size(k, N-1), size(v, N-1), size(v, N)
        k = Flux.unsqueeze(k, N-1)
        v = Flux.unsqueeze(v, N) .* k  # v*k' of size (spatial_dims..., c2, c1, b)
        v = _pad_to_size(v, padded_dims)
            # conv with precomputed window_spectrum = rfft(window)
        v = rfft_plan \ ((rfft_plan * v) .* window_spectrum)
        v = _unpad_to_size(v, radius, spatial_dims)
        out = dropdims(sum(v .* k; dims=N); dims=N)
    end
#=
    # fft-conv specific padding for cpu Arrays
    function _pad_to_size(arr::Array, ns)
        ns = _subst_to(ns, size(arr))
        out = zeros(eltype(arr), ns)
        cartidxs = CartesianIndices(arr)
        @inbounds out[cartidxs] .= arr
        return out
    end

    # fft-conv specific padding for gpu Arrays
    function _pad_to_size(arr::CuArray, ns)
        ns = _subst_to(ns, size(arr))
        out = CUDA.zeros(eltype(arr), ns)
        cartidxs = CartesianIndices(arr)
        @inbounds out[cartidxs] .= arr
        return out
    end
=#
    # fft-conv specific padding for cpu Arrays
    function _pad_to_size(arr::Array, ns)
        spdim = length(ns)
        ns = _diff_to(ns, size(arr))
        out = cat(arr, zeros(eltype(arr), ns...); dims = 1:spdim)
        return out
    end

    # fft-conv specific padding for gpu Arrays
    function _pad_to_size(arr::CuArray, ns)
        spdim = length(ns)
        ns = _diff_to(ns, size(arr))
        out = cat(arr, CUDA.zeros(eltype(arr), ns...); dims = 1:spdim)
        return out
    end

    # fft-conv specific unpadding
    function _unpad_to_size(arr, r, ns)
        nspatials = length(ns)
        ns = _subst_to(ns, size(arr))
        offset = ntuple(i -> (i <= nspatials ? r : zero(r)), length(ns))
        idx0 = CartesianIndex(offset)
        idxs = CartesianIndices(map(Base.OneTo, ns)) .+ idx0
        # returns a contiguous copy of subarray, not a view!
        return @inbounds arr[idxs]
    end

    @generated function _subst_to(v1::NTuple{N, T}, v2::NTuple{M, T}) where {N, T, M}
        M <= N && return :(v1[1:$M])
        elems = Vector{Expr}(undef, M)
        for i in 1:N
            elems[i] = :(v1[$i])
        end
        for i in N+1:M
            elems[i] = :(v2[$i])
        end
        return Expr(:tuple, elems...)
    end

    @generated function _diff_to(v1::NTuple{N, T}, v2::NTuple{M, T}) where {N, T, M}
        M <= N && return :(v1[1:$M] .- v2)
        elems = Vector{Expr}(undef, M)
        for i in 1:N
            elems[i] = :(v1[$i] - v2[$i])
        end
        for i in N+1:M
            elems[i] = :(v2[$i])
        end
        return Expr(:tuple, elems...)
    end

end #module
