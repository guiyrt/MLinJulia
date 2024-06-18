using MLinJulia, Test


include("python/torch/utils.jl")

@testset "Layer tests" begin
    include("python/torch/layers.jl")
end