using Flux, TensorCast, CUDA


struct SinusoidalPositionEmbeddings
    dim::Int
end

Flux.@layer SinusoidalPositionEmbeddings

function (spe::SinusoidalPositionEmbeddings)(time::AbstractArray)
    halfdim = spe.dim÷2 - 1

    freq = [0:halfdim...] .* -Float32(log(10_000) / halfdim) .|> exp |> fn_if(gpu, time isa CuArray)
    @cast pos[e,t] := freq[e] * time[t]

    vcat(sin.(pos), cos.(pos))
end