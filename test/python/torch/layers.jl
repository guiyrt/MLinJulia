using PyCall, MLinJulia, Flux, Test, CUDA

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pyModels = pyimport("scripts.models")


@testset "CylindricalConv" begin
    # Random input
    data, torchdata = rand32tensors(9, 16, 45, 3, 1)

    # CylindricalConvTranspose layers
    torchcc = pyModels.CylindricalConv(3, 16, kernel_size=(3, 3, 3), stride=1, padding=0)
    cc = CylindricalConv(torchcc)

    # Identical output
    @test cc(data) ≈ torchcc(torchdata) |> fromtorchtensor
end

@testset "CylindricalConvTranspose" begin
    # Random input
    data, torchdata = rand32tensors(2, 14, 12, 16, 1)

    # CylindricalConvTranspose layers
    torchcct = pyModels.CylindricalConvTranspose(16, 16, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=1)
    cct = CylindricalConvTranspose(torchcct)

    # Identical output
    @test cct(data) ≈ torchcct(torchdata) |> fromtorchtensor
end

@testset "LinearAttention" begin
    # Random input
    data, torchdata = rand32tensors(9, 16, 45, 16, 1)

    # LinearAttention layers
    torchla = pyModels.LinearAttention(16, n_heads=1, dim_head=32, cylindrical=true)
    la = LinearAttention(torchla)

    # Nearly identical output
    @test la(data) ≈ torchla(torchdata) |> fromtorchtensor
end