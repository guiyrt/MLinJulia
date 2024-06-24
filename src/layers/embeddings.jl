using Flux, TensorCast, CUDA


struct SinusoidalPositionEmbeddings
    dim_in::Int
    hidden::Dense
    out::Dense
end

function SinusoidalPositionEmbeddings(dim::Int)
    halfdim = dim÷2
    SinusoidalPositionEmbeddings(
        halfdim÷4,
        Dense(halfdim÷2 => halfdim, gelu),
        Dense(halfdim => halfdim)
    )
end

Flux.@layer SinusoidalPositionEmbeddings

function (spe::SinusoidalPositionEmbeddings)(time::AbstractArray)
    freq = [0:(spe.dim_in-1)...] .* -Float32(log(10_000) / (spe.dim_in-1)) .|> exp |> fn_if(gpu, time isa CuArray)
    @cast pos[e,t] := freq[e] * time[t]

    vcat(sin.(pos), cos.(pos)) |> spe.hidden |> spe.out
end


struct MlpEmbeddings
    in::Dense
    hidden::Dense
    out::Dense
end

function MlpEmbeddings(dim::Int)
    halfdim = dim÷2
    MlpEmbeddings(
        Dense(1 => halfdim÷2, gelu),
        Dense(halfdim÷2 => halfdim, gelu),
        Dense(halfdim => halfdim)
    )
end

Flux.@layer MlpEmbeddings

(mlp::MlpEmbeddings)(time::AbstractArray) = time |> (x -> @cast _[v, 1] := x[v]) |> mlp.in |> mlp.hidden |> mlp.out