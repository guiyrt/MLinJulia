using PyCall, MLinJulia, Flux, Test, CUDA

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pymodels = pyimport("scripts.models")


@testset "Cond U-net" begin
    # Random input
    data, torchdata = rand32tensors(9, 16, 45, 3, 128)
    c, torchc = rand32tensors(1, 128)
    t, torcht = rand32tensors(1, 128)

    torchunet = pymodels.CondUnet(cond_dim=128, out_dim=1, channels=3, layer_sizes=[16, 16, 16, 32], block_attn=true, mid_attn=true,
                                cylindrical=true, compress_Z= true, data_shape=[1, 3, 45, 16, 9], cond_embed= false, time_embed=false)
    unet = CondUnet(torchunet)

    @test unet(data, c, t) ≈ torchunet(torchdata, torchc, torcht) |> fromtorchtensor
end