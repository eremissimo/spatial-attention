import ChainRulesTestUtils
include("spatial_attention.jl")
using CUDA
using .SpatialAttention: MeanWindow
using Flux: params, withgradient


function timing_cpu(key::Array, val::Array, radius)
    nd = ndims(key)
    spatials, c1, c2, b = size(key)[1:nd-2], size(key, nd-1), size(val, nd-1), size(val, nd);
    window = MeanWindow{eltype(key), nd-2}(radius);
    model = SpatialAttention.Singlehead(spatials, b, c1=>6, c2=>7, 4, window)
    out, grads = withgradient(params(model)) do
        sum(model(key, val))
    end
    return out, grads
end

function timing_gpu(key::CuArray, val::CuArray, radius)
    nd = ndims(key)
    spatials, c1, c2, b = size(key)[1:nd-2], size(key, nd-1), size(val, nd-1), size(val, nd);
    window = MeanWindow{eltype(key), nd-2}(radius);
    model = SpatialAttention.Singlehead(spatials, b, c1=>6, c2=>7, 4, window) |> gpu
    out, grads = withgradient(params(model)) do
        sum(model(key, val))
    end
    return out, grads
end

function test_pad_rrule()
    arr = randn(Float64, 3,4,5,6)
    nd = (6,8)
    ChainRulesTestUtils.test_rrule(SpatialAttention._pad, arr, nd)
end
