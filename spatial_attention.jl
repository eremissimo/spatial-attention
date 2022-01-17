module SpatialAttention
    import Flux, FFTW, ChainRulesCore
    using CUDA
    include("window_functions.jl")
    using .WindowFunctions: AbstractWindow, MeanWindow, SumWindow, GaussianWindow

    export AbstractWindow, MeanWindow, SumWindow, GaussianWindow
    export Singlehead, Multihead


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

    # precomputed fft parameters (i.e. window_spectrum and rfft_plan)
    # are shared by all heads
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


    # ======== SINGLEHEAD ========= #

    function Singlehead(spatial_sizes, batch_size, sz_wkey,
        sz_wval, n_wout, window; σ_k=identity, σ=identity)
        # unfortunately we need to know spatial and batch sizes beforehand
        # to get an fft plan for v*k'. This makes input args list even messier.
        kershape = ntuple(i->1, length(spatial_sizes))
        wargs = map(chan -> Flux.convfilter(kershape, chan),
                        (sz_wkey, sz_wval, last(sz_wval) => n_wout))
        wtype = eltype(wargs[1])
        padded_sizes, radius, window_spectrum, rfft_plan = _init_fft(wtype,
                spatial_sizes, last(sz_wval), last(sz_wkey), batch_size, window)
        return Singlehead(wargs..., σ_k, σ, spatial_sizes, padded_sizes, radius,
                window_spectrum, rfft_plan)
    end

    Flux.@functor Singlehead (Wkey, Wval, Wout, window_spectrum, rfft_plan)

    Flux.trainable(self::Singlehead) = (self.Wkey, self.Wval, self.Wout)

    # 'move' cpu fft plan to gpu (reconstructing it from scratch actually)
    function Flux.adapt(::Flux.FluxCUDAAdaptor, pl::FFTW.rFFTWPlan{T}) where T
        tmp_array = CUDA.zeros(T, pl.sz...)
        gpu_pl = CUDA.CUFFT.plan_rfft(tmp_array, pl.region)
        return gpu_pl
    end

    # 'move' gpu fft plan to cpu (reconstructing it from scratch actually)
    function Flux.adapt(::Flux.FluxCPUAdaptor, pl:: CUDA.CUFFT.rCuFFTPlan{T}) where T
        tmp_array = zeros(T, pl.sz...)
        cpu_pl = FFTW.plan_rfft(tmp_array, pl.region)
        return cpu_pl
    end

    # Overriding Flux.gpu to make gpu() call compatible with fft plans.
    # BTW Flux.cpu fmaps everything, not only bits arrays, that's why there's no
    # need to implement the cpu counterpart.
    function Flux.gpu(x::Union{Singlehead, Multihead})
        Flux.check_use_cuda()
        Flux.use_cuda[] ? Flux.fmap(m -> Flux.adapt(Flux.FluxCUDAAdaptor(), m), x) : x
    end

    # forward function
    function (self::Singlehead)(key, val)
        k1 = self.σ_k.(Flux.conv(key, self.Wkey))
        v1 = Flux.conv(val, self.Wval)
        o1 = spa(k1, v1, self.window_spectrum, self.rfft_plan, self.spatial_sizes,
            self.padded_sizes, self.radius)
        out = self.σ.(Flux.conv(o1, self.Wout))
    end

    # pretty-print function
    function Base.show(io::IO, self::Singlehead{T, N}) where {T, N}
        println(io, "SpatialAttention.Singlehead{$T, $(N-2)D}[
                    radius=$(self.radius),
                    spatial_sizes=$(self.spatial_sizes)...]")
    end

    # the main computations of spatial attention layers
    function spa(k::AbstractArray{T, N}, v::AbstractArray{T,N},
        window_spectrum, rfft_plan, spatial_dims, padded_dims, radius) where {T, N}
        c1, c2, b = size(k, N-1), size(v, N-1), size(v, N)
        k = Flux.unsqueeze(k, N-1)
        v = Flux.unsqueeze(v, N) .* k  # v*k' of size (spatial_dims..., c2, c1, b)
        v = _pad(v, padded_dims)
            # conv with precomputed window_spectrum = rfft(window)
        v = rfft_plan \ ((rfft_plan * v) .* window_spectrum)
        v = _unpad(v, radius, spatial_dims)
        out = dropdims(sum(v .* k; dims=N); dims=N)
    end

    # ======= MULTIHEAD ======= #

    function Multihead(nheads, spatial_sizes, batch_size, sz_wkey,
        sz_wval, n_wout, window; σ_k=identity, σ=identity)
        kershape = ntuple(i->1, length(spatial_sizes))
        wargs = map((sz_wkey, sz_wval, last(sz_wval) => n_wout)) do chan
            ntuple(j -> Flux.convfilter(kershape, chan), nheads)
        end
        wtype = eltype(wargs[1][1])
        padded_sizes, radius, window_spectrum, rfft_plan = _init_fft(wtype,
                spatial_sizes, last(sz_wval), last(sz_wkey), batch_size, window)
        return Multihead(wargs..., σ_k, σ, spatial_sizes, padded_sizes, radius,
                window_spectrum, rfft_plan)
    end

    Flux.@functor Multihead (Wkey, Wval, Wout, window_spectrum, rfft_plan)

    Flux.trainable(self::Multihead) = (self.Wkey, self.Wval, self.Wout)

    # forward function (mapreduce variant)
    function (self::Multihead)(key, val)
        return mapreduce(+, zip(self.Wkey, self.Wval, self.Wout)) do (Wk, Wv, Wo)
            k1 = self.σ_k.(Flux.conv(key, Wk))
            v1 = Flux.conv(val, Wv)
            o1 = spa(k1, v1, self.window_spectrum, self.rfft_plan, self.spatial_sizes,
                self.padded_sizes, self.radius)
            out = self.σ.(Flux.conv(o1, Wo))
        end
    end

    # pretty-print function
    function Base.show(io::IO, self::Multihead{T, N, M}) where {T, N, M}
        println(io, "SpatialAttention.Multihead{$T, $(N-2)D}[
                    nheads=$M,
                    radius=$(self.radius),
                    spatial_sizes=$(self.spatial_sizes)...]")
    end

    # ======= HELPERS ======= #

    function _init_fft(type, spatial_sizes, vchan, kchan, batch_size, window)
        sp_nd = length(spatial_sizes)
        radius = WindowFunctions.get_radius(window)
        # nextprod is for faster fft
        padded_sizes = map(d->nextprod([2,3,5], d), spatial_sizes .+ radius)
        kernel_padded = _pad(WindowFunctions.construct_kernel(window, sp_nd, type),
                            padded_sizes)
        window_spectrum = FFTW.rfft(kernel_padded)
        # constructing fft plan for v*k' for a fast convolution with the window.
        # It is possible to just use NNlib.conv for the purpose of convolving
        # with the predefined kernel and not mess around with paddings and ffts
        # but I think that would be slower for large window radii.
        # I'll check that later (<- TODO)
        tmp_vkT = randn(type, padded_sizes..., vchan, kchan, batch_size)
        rfft_plan = FFTW.plan_rfft(tmp_vkT, 1:sp_nd)
        return (padded_sizes, radius, window_spectrum, rfft_plan)
    end

    # fft-conv specific padding for cpu Arrays
    function _pad(arr::Array, ns)
        spdim = length(ns)
        ns = _diff_to(ns, size(arr))
        out = cat(arr, zeros(eltype(arr), ns...); dims = 1:spdim)
        return out
    end

    # fft-conv specific padding for gpu Arrays
    function _pad(arr::CuArray, ns)
        spdim = length(ns)
        ns = _diff_to(ns, size(arr))
        out = cat(arr, CUDA.zeros(eltype(arr), ns...); dims = 1:spdim)
        return out
    end

    # Zygote couldn't differentiate through _pad because of missing CUDA.zeros adjoint
    # I could add the missing adjoint... but nah, lets implement the whole
    # differentiation rule instead :P
    function ChainRulesCore.rrule(::typeof(_pad), arr, ns)
        orig_shape = size(arr)
        out = _pad(arr, ns)
        function pad_pullback(∇out)
            return(ChainRulesCore.NoTangent(),
                ChainRulesCore.@thunk(@inbounds view(ChainRulesCore.unthunk(∇out),
                                                Base.OneTo.(orig_shape)...)),
                ChainRulesCore.NoTangent())
        end
        return out, pad_pullback
    end

    # fft-conv specific unpadding
    function _unpad(arr, r, ns)
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
