using CUDA, Base.Cartesian


# Extension of Cartesian.nref that makes possible partial ref generation
# like A[i_1, i_2, i_3, k, b] for predefined expression tuple rest = (k, b)
# Usage: @nref n A i (k, b) is equivalent of A[i_1, i_2, ..., i_n, k, b]
macro _nref(N::Int, A::Symbol, ex::Symbol, rest)
    vars = Any[ Cartesian.inlineanonymous(ex,i) for i = 1:N ]
    return Expr(:escape, Expr(:ref, A, vars..., rest.args...))
end

# macro for sum expr generation
macro _nsum(N::Int, summand::Expr)
    exprs = Any[ Expr(:escape, Cartesian.inlineanonymous(summand, i)) for i = 1:N ]
    return Expr(:call, :+, exprs...)
end

# Woah! Important note:
# https://github.com/JuliaLang/julia/issues/21356#issuecomment-293467627

#======== STENCIL LOOPS =========#

function __spa_stencil_loop_expr(T, N, kchan, vchan)
    return quote
        # key[I...,:,B]
        @inbounds kI = @ntuple $kchan k -> @_nref($(N-2), key, I, (k, B))
        @nexprs $vchan k -> oI_k = zero($T)
        @inbounds @nloops $(N-2) i n->max(1, I_n - radius):min(size(key, n), I_n + radius) begin
            # dott = sum(key[I...,k,B]*key[i...,k,B] for k in 1:kchan)
            dott = @_nsum $kchan k -> kI[k]*@_nref($(N-2), key, i, (k, B))
            # out[I...,k,B] += dott*val[i...,k,B]
            @nexprs $vchan k -> oI_k += dott*@_nref($(N-2), val, i, (k, B))
        end
        @inbounds @nexprs $vchan k -> @_nref($(N-2), out, I, (k, B)) = oI_k*normalizer
    end
end


function __∇key_stencil_loop_expr(T, N, kchan, vchan)
    return quote
        # val[I...,:,B]
        @inbounds vI = @ntuple $vchan k -> @_nref($(N-2), val, I, (k, B))
        # ∇out[I...,:,B]
        @inbounds doI = @ntuple $vchan k -> @_nref($(N-2), ∇out, I, (k, B))
        @nexprs $kchan k -> dkI_k = zero($T)
        @inbounds @nloops $(N-2) i n->max(1, I_n - radius):min(size(key, n), I_n + radius) begin
            # dott = sum( ∇out[i...,k,B]*val[I...,k,B] + ∇out[I...,k,B]*val[i...,k,B]
            #           for k in 1:kchan)
            dott = @_nsum $vchan k -> (@_nref($(N-2), ∇out, i, (k, B)) * vI[k] +
                                doI[k] * @_nref($(N-2), val, i, (k, B)) )
            # ∇key[I...,k,B] += dott*key[i...,k,B]
            @nexprs $kchan k -> dkI_k += dott*@_nref($(N-2), key, i, (k, B))
        end
        @inbounds @nexprs $kchan k -> @_nref($(N-2), ∇key, I, (k, B)) = dkI_k*normalizer
    end
end


function __∇val_stencil_loop_expr(T, N, kchan, vchan)
    return quote
        # key[I...,:,B]
        @inbounds kI = @ntuple $kchan k -> @_nref($(N-2), key, I, (k, B))
        @nexprs $vchan k -> dvI_k = zero($T)
        @inbounds @nloops $(N-2) i n->max(1, I_n - radius):min(size(key, n), I_n + radius) begin
            # dott = sum(key[I...,k,B]*key[i...,k,B] for k in 1:kchan)
            dott = @_nsum $kchan k -> kI[k]*@_nref($(N-2), key, i, (k, B))
            # ∇val[I...,k,B] += dott*∇out[i...,k,B]
            @nexprs $vchan k -> dvI_k += dott*@_nref($(N-2), ∇out, i, (k, B))
        end
        @inbounds @nexprs $vchan k -> @_nref($(N-2), ∇val, I, (k, B)) = dvI_k*normalizer
    end
end


#======= CUDA KERNELS =======#

@generated function spa_kernel!(out::CuDeviceArray{T, N},
                                key::CuDeviceArray{T, N},
                                val::CuDeviceArray{T, N},
                                radius, normalizer, ::Val{K}, ::Val{V}) where {T, N, K, V}
    return quote
        # assignments of indices.
        # TODO: Note that this part of code
        # is not general but for my use cases it's ok. This issue is
        # manageble by proper generalization of assignment between pixel index
        # and thread/block indices.
        # For now it's left as follows: fitst spatial dim is threadIdx.x,
        # second spatial dim (if exists) is blockIdx.x, then goes blockIdx.y etc.
        # Last blockIdx is batch dim.
        I_1 = threadIdx().x
        @nexprs $(N-3) (d -> I_{d+1} = blockIdx()[d])
        B = blockIdx()[$(N-2)]
        # main loop
        $(__spa_stencil_loop_expr(T, N, K, V))
        return nothing
    end
end

@generated function ∇key_kernel!(∇key::CuDeviceArray{T, N},
                                ∇out::CuDeviceArray{T, N},
                                key::CuDeviceArray{T, N},
                                val::CuDeviceArray{T, N},
                                radius, normalizer, ::Val{K}, ::Val{V}) where {T, N, K, V}
    return quote
        # assignments of indices.
        I_1 = threadIdx().x
        @nexprs $(N-3) (d -> I_{d+1} = blockIdx()[d])
        B = blockIdx()[$(N-2)]
        # main loop
        $(__∇key_stencil_loop_expr(T, N, K, V))
        return nothing
    end
end

@generated function ∇val_kernel!(∇val::CuDeviceArray{T, N},
                                ∇out::CuDeviceArray{T, N},
                                key::CuDeviceArray{T, N},
                                val::CuDeviceArray{T, N},
                                radius, normalizer, ::Val{K}, ::Val{V}) where {T, N, K, V}
    return quote
        # assignments of indices.
        I_1 = threadIdx().x
        @nexprs $(N-3) (d -> I_{d+1} = blockIdx()[d])
        B = blockIdx()[$(N-2)]
        # main loop
        $(__∇val_stencil_loop_expr(T, N, K, V))
        return nothing
    end
end


#====== CPU 'KERNELS' =======#

@generated function spa_kernel!(out::Array{T, N},
                                key::Array{T, N},
                                val::Array{T, N},
                                radius, normalizer, ::Val{K}, ::Val{V}) where {T, N, K, V}
    return quote
        for B in 1:size(key, $N)
            @nloops $(N-2) I key begin
                $(__spa_stencil_loop_expr(T, N, K, V))
            end  # for I_1...I_N-2
        end   # for B
        return nothing
    end
end

@generated function ∇key_kernel!(∇key::Array{T, N},
                                ∇out::Array{T, N},
                                key::Array{T, N},
                                val::Array{T, N},
                                radius, normalizer, ::Val{K}, ::Val{V}) where {T, N, K, V}
    return quote
        for B in 1:size(key, $N)
            @nloops $(N-2) I key begin
                $(__∇key_stencil_loop_expr(T, N, K, V))
            end  # for I_1...I_N-2
        end   # for B
        return nothing
    end
end

@generated function ∇val_kernel!(∇val::Array{T, N},
                                ∇out::Array{T, N},
                                key::Array{T, N},
                                val::Array{T, N},
                                radius, normalizer, ::Val{K}, ::Val{V}) where {T, N, K, V}
    return quote
        for B in 1:size(key, $N)
            @nloops $(N-2) I key begin
                $(__∇val_stencil_loop_expr(T, N, K, V))
            end  # for I_1...I_N-2
        end   # for B
        return nothing
    end
end
