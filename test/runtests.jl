using MLinJulia, Test


include("python/torch/utils.jl")

@testset "CaloDiffusion" begin
    include("python/calodiffusion.jl")
end

@testset "Data" begin
    include("python/data.jl")
end

@testset "Torch layers" begin
    include("python/torch/base.jl")
end

@testset "Custom layers" begin
    include("python/torch/layers.jl")
end

@testset "Models" begin
   include("python/torch/models.jl") 
end