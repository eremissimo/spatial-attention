module SpatialAttentionUnnormalized

    using CUDA
    import ChainRulesCore, Flux
    include(".\\kernels_unnorm.jl")

    export SingleheadSpatialAttention, MultiheadSpatialAttention

    const PorT = Union{Pair{Int, Int}, Tuple{Int, Int}}

    struct SingleheadSpatialAttention
        Wkey
        Wval
        Wout
        σ_k
        σ
        radius::Int
    end

    struct MultiheadSpatialAttention
        connection
        layers
    end


    """
        spa(key::CuArray, val::CuArray, radius::Int)

    CUDA version of core forward function for SingleheadSpatialAttention
    """
    function spa(key::CuArray{T, N}, val::CuArray{T, N}, radius::Int) where {T, N}
        sz1 = size(key, 1)
        sz2n = (size(key)[2:N-2]..., size(key, N))
        kchan = size(key, N-1)
        vchan = size(val, N-1)
        normalizer = one(T)/(kchan*(2*radius + 1)^2)
        out = CUDA.zero(val)
        @cuda threads=sz1 blocks=sz2n spa_kernel!(out, key, val, radius,
                                    normalizer, Val(kchan), Val(vchan))
        return out
    end

    """
        spa(key::Array, val::Array, radius::Int)

    CPU version of core forward function for SingleheadSpatialAttention
    """
    function spa(key::Array{T, N}, val::Array{T, N}, radius::Int) where {T, N}
        kchan = size(key, N-1)
        vchan = size(val, N-1)
        normalizer = one(T)/(kchan*(2*radius + 1)^2)
        out = zero(val)
        spa_kernel!(out, key, val, radius, normalizer, Val(kchan), Val(vchan))
        return out
    end


    """
        SingleheadSpatialAttention(nd, in_key => out_key, in_val => out_val, out_out; radius=1, σ_k=id, σ=id)

    Local 2D spatial self attention layer with unnormalized key similarity coefficients
    A kind of normalization is possible with key activation σ_k = sigmoid.
    At least it limits the possible range of keys dot product
    Args:
        nd: number of spatial dimensions of tensors (1 for 1d, 2 for 2d etc.)
        in_key => out_key: transform of key channels
        in_val => out_val: transform of value channels
        out_out: number of output channels
    Kwargs:
        radius: local neighborhood of attention at every pixel
        σ_k: key activation function (for normalization)
        σ: output activation function
    """
    function SingleheadSpatialAttention(nd::Int, sz_wkey::PorT,
        sz_wval::PorT, n_wout::Int; radius=1, σ_k=identity, σ=identity)
        wargs = map((sz_wkey, sz_wval, last(sz_wval) => n_wout)) do chan
            Flux.convfilter(ntuple(i->1, nd), chan)
        end
        return SingleheadSpatialAttention(wargs..., σ_k, σ, radius)
    end

    #forward
    function (self::SingleheadSpatialAttention)(key, val)
        k1 = self.σ_k.(Flux.conv(key, self.Wkey))
        v1 = Flux.conv(val, self.Wval)
        o1 = spa(k1, v1, self.radius)
        out = self.σ.(Flux.conv(o1, self.Wout))
        return out
    end

    """
        MultiheadSpatialAttention(connection, nheads, att_args...; att_kwargs...)

    Local multihead spatial self attention layer with unnormalized key similarity coefficients
    Args:
        connection: a binary function of output reduction between attention heads.
                    Usually it is '+'.
        nheads: number of heads
        att_args...: args for single heads initialization
        att_kwargs...: kwargs for single heads initialization
    """
    function MultiheadSpatialAttention(connection, nheads, args...; kwargs...)
        layers = ntuple(i->SingleheadSpatialAttention(args...; kwargs...), nheads)
        return MultiheadSpatialAttention(connection, layers)
    end

    #forward
    (self::MultiheadSpatialAttention)(k, v) = mapreduce(f -> f(k, v), self.connection, self.layers)
    # Note that Flux.Parallel won't work here as expected. It broadcasts non single input
    # through branches. So it produces connection(layers[1](k), layers[2](v)).

    Flux.@functor SingleheadSpatialAttention (Wkey, Wval, Wout)
    Flux.@functor MultiheadSpatialAttention

    function Base.show(io::IO, self::SingleheadSpatialAttention)
        nd = ndims(self.Wkey)-2      # TODO: probably it would be better
                                    # to parametrize the type with parameters' order
        println(io, "SingleheadSpatialAttention{$(nd)D}[radius=$(self.radius),...]")
    end

    function Base.show(io::IO, self::MultiheadSpatialAttention)
        nd = ndims(self.layers[1].Wkey)-2    # TODO: yeah this too
        n = length(self.layers)
        println(io, "MultiheadSpatialAttention{$(nd)D}[nheads=$n, radius=$(self.layers[1].radius),...]")
    end

    ## Gradients

    # key
    function ∇key_spa!(∇key::CuArray{T, N}, ∇out::CuArray{T, N}, key::CuArray{T, N},
                        val::CuArray{T, N}, radius::Int) where {T, N}
        sz1 = size(key, 1)
        sz2n = (size(key)[2:N-2]..., size(key, N))
        kchan = size(key, N-1)
        vchan = size(val, N-1)
        normalizer = one(T)/(kchan*(2*radius + 1)^2)
        @cuda threads=sz1 blocks=sz2n ∇key_kernel!(∇key, ∇out, key, val, radius,
                                            normalizer, Val(kchan), Val(vchan))
        return nothing
    end

    function ∇key_spa(∇out::CuArray, key::CuArray, val::CuArray, radius::Int)
        ∇key = CUDA.zero(key)
        ∇key_spa!(∇key, ∇out, key, val, radius)
        return ∇key
    end

    function ∇key_spa!(∇key::Array{T, N}, ∇out::Array{T, N}, key::Array{T, N},
                        val::Array{T, N}, radius::Int) where {T, N}
        kchan = size(key, N-1)
        vchan = size(val, N-1)
        normalizer = one(T)/(kchan*(2*radius + 1)^2)
        ∇key_kernel!(∇key, ∇out, key, val, radius, normalizer, Val(kchan), Val(vchan))
        return nothing
    end

    function ∇key_spa(∇out::Array, key::Array, val::Array, radius::Int)
        ∇key = zero(key)
        ∇key_spa!(∇key, ∇out, key, val, radius)
        return ∇key
    end

    # val
    function ∇val_spa!(∇val::CuArray{T, N}, ∇out::CuArray{T, N}, key::CuArray{T, N},
                    val::CuArray{T, N}, radius::Int) where {T, N}
        sz1 = size(key, 1)
        sz2n = (size(key)[2:N-2]..., size(key, N))
        kchan = size(key, N-1)
        vchan = size(val, N-1)
        normalizer = one(T)/(kchan*(2*radius + 1)^2)
        @cuda threads=sz1 blocks=sz2n ∇val_kernel!(∇val, ∇out, key, val, radius,
                                            normalizer, Val(kchan), Val(vchan))
        return nothing
    end

    function ∇val_spa(∇out::CuArray, key::CuArray, val::CuArray, radius::Int)
        ∇val = CUDA.zero(val)
        ∇val_spa!(∇val, ∇out, key, val, radius)
        return ∇val
    end

    function ∇val_spa!(∇val::Array{T, N}, ∇out::Array{T, N}, key::Array{T, N},
                        val::Array{T, N}, radius::Int) where {T, N}
        kchan = size(key, N-1)
        vchan = size(val, N-1)
        normalizer = one(T)/(kchan*(2*radius + 1)^2)
        ∇val_kernel!(∇val, ∇out, key, val, radius, normalizer, Val(kchan), Val(vchan))
        return nothing
    end

    function ∇val_spa(∇out::Array, key::Array, val::Array, radius::Int)
        ∇val = zero(val)
        ∇val_spa!(∇val, ∇out, key, val, radius)
        return ∇val
    end


    """
        rrule(::typeof(spa2d), key::AbstractArray, val::AbstractArray; radius=1)

    Lazy variant of spa differentiation rule for Zygote AD
    """
    function ChainRulesCore.rrule(::typeof(spa), key, val, radius)
        out = spa(key, val, radius)
        function spa_pullback(∇outth)
            ∇out = ChainRulesCore.unthunk(∇outth)
            return (ChainRulesCore.NoTangent(),
                    ChainRulesCore.@thunk(∇key_spa(∇out, key, val, radius)),
                    ChainRulesCore.@thunk(∇val_spa(∇out, key, val, radius)),
                    ChainRulesCore.NoTangent())
        end
        return out, spa_pullback
    end

    #=
    """
        rrule(::typeof(spa), key::AbstractArray, val::AbstractArray; radius=1)

    Eager variant of spa2d differentiation rule for Zygote AD
    """
    function ChainRulesCore.rrule(::typeof(spa), key, val, radius=1)
        out = spa(key, val, radius)
        function spa_pullback(∇outth)
            ∇out = ChainRulesCore.unthunk(∇outth)
            ∇key = zero(key)
            ∇val = zero(val)
            @sync begin
                @async begin
                    ∇key_spa!(∇key, ∇out, key, val, radius)
                    nothing
                end
                @async begin
                    ∇val_spa!(∇val, ∇out, key, val, radius)
                    nothing
                end
            end
            return (ChainRulesCore.NoTangent(), ∇key, ∇val, ChainRulesCore.NoTangent())
        end
        return out, spa_pullback
    end
    =#


end  #module
