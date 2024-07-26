using Underscores

"""
Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
"""
struct CosineSchedule
    β::Vector{Float32}
    α::Vector{Float32}
    sqrt_α⁻¹::Vector{Float32}
    sqrt_αcumprod::Vector{Float32}
    sqrt_one_minus_αcumprod::Vector{Float32}
    posterior_variance::Vector{Float32}

    function CosineSchedule(nsteps::Int; s::Float32 = 0.008f0)
        steps = LinRange(0f0, nsteps, nsteps+1)
        alphas_cumprod = cos.(((steps ./ nsteps) .+ s ) ./ (1.0f0 + s) .* 0.5f0π) .^ 2.0f0
        alphas_cumprod = alphas_cumprod ./ alphas_cumprod[1]
        β = clamp.(1f0 .- (alphas_cumprod[2:end] ./ alphas_cumprod[1:end-1]), 0.0001f0, 0.9999f0)
        
        α = 1.0f0 .- β
        αcumprod = cumprod(α)

        new(β, α,
            sqrt.(1.0f0 ./ α),
            sqrt.(αcumprod),
            sqrt.(1.0f0 .- αcumprod),
            β .* (1.0f0 .- [1.0f0; αcumprod[1:end-1]]) ./ (1.0f0 .- αcumprod)
        )
    end
end