include("window_functions.jl")
include("spatial_attention.jl")
using .WindowFunctions
using Zygote: withgradient
using Flux: params

function test1_cpu()
    key = randn(Float32, 10, 20, 15, 3, 5)
    val = randn(Float32, 10, 20, 15, 6, 5)
    spatials, c1, c2, b = size(key)[1:3], size(key, 4), size(val, 4), size(val, 5)
    window = MeanWindow{Float32, 3}(2)
    model = SpatialAttention.Singlehead(spatials, b, 3=>6, 6=>7, 4, window)
    out, grads = withgradient(params(model)) do
        sum(model(key, val))
    end
    return out, grads
end
