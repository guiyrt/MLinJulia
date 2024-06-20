using PyCall, MLinJulia, Flux, Test, CUDA

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pymodels = pyimport("scripts.models")


@testset "CylindricalConv" begin
    # Random input
    data, torchdata = rand32tensors(9, 16, 45, 3, 1)

    # CylindricalConv layers
    torchcc = pymodels.CylindricalConv(3, 16, kernel_size=(3, 3, 3), stride=1, padding=0)
    cc = CylindricalConv(torchcc)

    # Identical output
    @test cc(data) ≈ torchcc(torchdata) |> fromtorchtensor
end

@testset "CylindricalConvTranspose" begin
    # Random input
    data, torchdata = rand32tensors(2, 14, 12, 16, 1)

    # CylindricalConvTranspose layers
    torchcct = pymodels.CylindricalConvTranspose(16, 16, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=1)
    cct = CylindricalConvTranspose(torchcct)

    # Identical output
    @test cct(data) ≈ torchcct(torchdata) |> fromtorchtensor
end

@testset "LinearAttention" begin
    # Random input
    data, torchdata = rand32tensors(9, 16, 45, 16, 1)

    # LinearAttention layers
    torchla = pymodels.LinearAttention(16, n_heads=1, dim_head=32, cylindrical=true)
    la = LinearAttention(torchla)

    # Nearly identical output
    @test la(data) ≈ torchla(torchdata) |> fromtorchtensor
end

@testset "Residual PreNorm LinearAttention" begin
    # Random input
    data, torchdata = rand32tensors(9, 16, 45, 16, 1)

    # LinearAttention layers
    torchres = pymodels.Residual(pymodels.PreNorm(16, pymodels.LinearAttention(16, n_heads=1, dim_head=32, cylindrical=true)))
    res = Residual(torchres)

    # Nearly identical output
    @test res(data) ≈ torchres(torchdata) |> fromtorchtensor
end

@testset "SinusoidalPositionEmbeddings" begin
    data, torchdata = randinttensors([0:256...], 128)

    torchspe = pymodels.SinusoidalPositionEmbeddings(32)
    spe = SinusoidalPositionEmbeddings(torchspe)
    
    @test spe(data) ≈ torchspe(torchdata) |> fromtorchtensor
end