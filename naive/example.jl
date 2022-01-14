using CUDA, Flux, MLDatasets
using Flux.Data: DataLoader
using Flux.Losses: logitcrossentropy
using Flux.Optimise: update!, ADAM
using ParameterSchedulers: Exp
using Printf: @sprintf
using ProgressBars: ProgressBar, set_description
include(".\\spatial_attention_dumb.jl")
using .SpatialAttentionUnnormalized


function get_data(batch_size)
    train_x, train_y = MNIST.traindata(Float32);
    return DataLoader((train_x, train_y); batchsize=batch_size, shuffle=true, partial=false)
end

#model
struct SingleSelfAttentionBlock
    key
    val
    sa
    head
end

#forward
function (self::SingleSelfAttentionBlock)(x)
    x = Flux.maxpool(x, (2,2))               # to increase receptive field
    k = self.key(x)
    v = self.val(x)
    out = self.sa(k,v) |> self.head
    out = dropdims(out; dims=(1,2))      # aka squeeze
    return out
end

Flux.@functor SingleSelfAttentionBlock

# Attention is all you need for MNIST
# (Not really. Much simpler convnets converge a lot faster)
function construct_model(;nheads=3, key_channels=3, val_channels=4, radius=5)
    nd = 2               # 2D images
    out_channels = 10    # 0...9
    key = Flux.Conv((5,5), 1=>key_channels, relu; pad=1)
    val = Flux.Conv((5,5), 1=>val_channels, relu; pad=1)
    sa = MultiheadSpatialAttention(+, nheads, nd, key_channels=>key_channels,
            val_channels=>val_channels, out_channels; σ_k=identity, σ=mish, radius=radius)
    head = GlobalMeanPool()
    return SingleSelfAttentionBlock(key, val, sa, head)
end

function train!(model, optimizer, scheduler, dataloader, epochs, device)
    model = model |> device
    ps = params(model)
    bar_loader = ProgressBar(dataloader, printing_delay=0.5)
    unsqueezer = Flux.unsqueeze(3)
    for (η, epoch) in zip(scheduler, 1:epochs)
        optimizer.eta = η
        for (imgs, targs) in bar_loader
            imgs = imgs |> unsqueezer|> device
            targets = Flux.onehotbatch(targs, 0:9) |> device
            loss, grads = Flux.withgradient(ps) do
                logits = model(imgs)
                logitcrossentropy(logits, targets)
            end
            update!(optimizer, ps, grads)
            set_description(bar_loader, @sprintf("%d) Loss: %.2f", epoch, loss))
        end
    end
    return nothing
end

function main(epochs; lr=1f-2, batch_size=500)
    device = CUDA.functional() ? gpu : cpu
    optimizer = ADAM(lr)
    scheduler = Exp(; λ=lr, γ=0.8f0)
    model = construct_model()
    dataloader = get_data(batch_size)
    train!(model, optimizer, scheduler, dataloader, epochs, device)
    return nothing
end
