using NVTX
using Statistics
import ChainRulesCore: @ignore_derivatives

function batchloss(m::CondUnet, c::TrainingConfig, x::AbstractArray, e::AbstractArray, t::AbstractArray, noise::AbstractArray)
    @ignore_derivatives NVTX.@mark "batchloss: t_emb"
    t_emb = reshape(c.sched.sqrt_one_minus_αcumprod[t], (1, :))

    @ignore_derivatives NVTX.@mark "batchloss: apply noise"
    xₜ = applynoise(c.sched, x, noise, t)

    @ignore_derivatives NVTX.@mark "batchloss: run model"
    x̂ = @_ xₜ |> cat(__, repeat(c.pos_images, 1, 1, 1, 1, size(x)[end]); dims=4) |> m(__, e, t_emb)

    @ignore_derivatives NVTX.@mark "batchloss: calculate loss"
    if c.noise_pred_loss
        loss = Flux.mse(noise, x̂)
    else
        σ = extract(c.sched.sqrt_one_minus_αcumprod, t) .^ 2f0
        c_skip = 1f0 ./ (σ .+ 1f0)
        c_out = 1f0 ./ sqrt.(1f0 .+ (1f0 ./ σ))
        weight = 1f0 .+ (1f0 ./ σ)

        loss = sum(weight .* ((c_skip .* xₜ + c_out .* x̂ .- x) .^ 2f0)) / (mean(weight) * c.calo.nvoxels)
    end

    @ignore_derivatives NVTX.@mark "batchloss: end"
end