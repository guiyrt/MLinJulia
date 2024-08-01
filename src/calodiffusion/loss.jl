function noise_weighthed_avg_loss(x::AbstractArray, x̂::AbstractArray, t_emd::AbstractArray, nvoxels::Int)
    weight = 1f0 + 1f0 ./ (t_emd .^ 2f0)
    return sum(weight .* ((x̂ .- x) .^ 2f0)) ./ (mean(weight) * nvoxels)
end


function batchloss(m::CondUnet, c::TrainingConfig, x::AbstractArray, e::AbstractArray, t::AbstractArray, noise::AbstractArray)
    t_emb = reshape(c.sched.sqrt_one_minus_αcumprod[t], (1, :))

    xₜ = applynoise(c.sched, x, noise, t)

    x̂ = @_ xₜ |> cat(__, repeat(c.pos_images, 1, 1, 1, 1, size(x)[end]); dims=4) |> m(__, e, t_emb)

    c.noise_pred_loss ? Flux.mse(noise, x̂) : noise_weighthed_avg_loss(x, x̂, extract(c.sched.sqrt_one_minus_αcumprod, t), c.calo.nvoxels)
end