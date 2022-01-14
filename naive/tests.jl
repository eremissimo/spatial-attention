using Random, CUDA
import ChainRulesTestUtils
#import FiniteDifferences
using Flux: sigmoid, gpu, cpu, gradient, params, fmap
using Zygote: withgradient
using Base.Threads: @spawn, @sync
using Test: @testset, @test
include(".\\spatial_attention_dumb.jl")


function mdims(chan, batch, ndim)
    dims = map(i -> 5 + i, 1:ndim)
    push!(dims, chan)
    push!(dims, batch)
    return Tuple(dims)
end

# extension for compatibility
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::CuArray)
    return 9 .* CUDA.randn(size(x)...)
end

#=
# extension for compatibility
function FiniteDifferences.to_vec(x::CuArray)
    shp = size(x)
    vec = x[:]
    function back_from_vec(v::CuArray)
        reshape(v, shp)
    end
    return vec, back_from_vec
end

# extension for compatibility
function FiniteDifferences._j′vp(fdm, f, ȳ::CuArray{T, 1}, x::CuArray{T, 1}) where T
    isempty(x) && return CuArray{T, 1}[] # if x is empty, then so is the jacobian and x̄
    return transpose(first(FiniteDifferences.jacobian(fdm, f, x))) * ȳ
end
=#

# This fails due to some incompatibility between FiniteDifferences
# internals (to_vec, _j'vp, jacobian etc.) and CUDA.
function test_cuda_gradrule(ndim)
    radius = 2
    key = CUDA.randn(mdims(4, 3, ndim)...)
    val = CUDA.randn(mdims(6, 3, ndim)...)
    ChainRulesTestUtils.test_rrule(SpatialAttentionUnnormalized.spa, key, val, radius)
end

function test_cpu_gradrule(ndim)
    radius = 2
    key = randn(mdims(4, 3, ndim)...)
    val = randn(mdims(6, 3, ndim)...)
    ChainRulesTestUtils.test_rrule(SpatialAttentionUnnormalized.spa, key, val, radius)
end

function timing_grads_singlehead(key, val, radius=2)
    nd = ndims(key)
    channel_dim = nd - 1;
    kch = size(key, channel_dim)
    vch = size(val, channel_dim)
    sa = SpatialAttentionUnnormalized.SingleheadSpatialAttention(nd-2, kch=>4,
                                        vch=>7, 3; radius=radius, σ_k=sigmoid)
    if isa(key, CuArray)
        sa = gpu(sa)
    end
    out, grads = withgradient(params(sa)) do
        sum(sa(key, val))
    end
    return out, grads
end

function timing_grads_multihead(key, val, radius=2)
    nheads = 3
    nd = ndims(key)
    channel_dim = nd - 1;
    kch = size(key, channel_dim)
    vch = size(val, channel_dim)
    sa = SpatialAttentionUnnormalized.MultiheadSpatialAttention(+, nheads, nd-2,
                                kch=>4, vch=>7, 3; radius=radius, σ_k=sigmoid)
    if isa(key, CuArray)
        sa = gpu(sa)
    end
    out, grads = withgradient(params(sa)) do
        sum(sa(key, val))
    end
    return out, grads
end

function test_cpu_eq_gpu(n, radius)
    key = randn(Float32, n,n,n, 4, 3)
    val = randn(Float32, n,n,n, 6, 3)
    sa = SpatialAttentionUnnormalized.SingleheadSpatialAttention(3, 4=>4, 6=>7,
                                        3; radius=radius, σ_k=sigmoid)
    key_gpu, val_gpu, sa_gpu = map(gpu, (key, val, sa))
    tsk = @spawn withgradient(params(sa)) do
        sum(sa(key, val))
    end
    out_gpu, grads_gpu = withgradient(params(sa_gpu)) do
        sum(sa_gpu(key_gpu, val_gpu))
    end
    out_gpu, grads_gpu... = map(cpu, (out_gpu, values(grads_gpu)...))
    out, grads = fetch(tsk)
    @testset "Test if cpu and gpu forward pass tensors are equal" begin
        @test out ≈ out_gpu
    end
    @testset "Test if cpu and gpu gradients are equal" begin
        for (g, gg) in zip(values(grads), grads_gpu)
            @test g ≈ gg
        end
    end
    return nothing
end


""" Run all tests. Verify differentiation rules on cpu, then check if gpu version
produces the same output and gradients as cpu version of spa layer"""
function run_tests()
    @sync begin
        @spawn test_cpu_gradrule(1)     # 1D
        @spawn test_cpu_gradrule(2)     # 2D
        @spawn test_cpu_gradrule(3)     # 3D
        test_cpu_eq_gpu(5, 2)
    end
    return nothing
end
