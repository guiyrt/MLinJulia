using PyCall, MLinJulia, Flux, Test, CUDA

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pymodels = pyimport("scripts.models")


@testset "CaloDiffusion" begin
    # Random input
    data, torchdata = rand32tensors(9, 16, 45, 3, 128)
    c, torchc = rand32tensors(1)
    t, torcht = rand32tensors(1)

    torchcd = pymodels.CondUnet(cond_dim=128, out_dim=1, channels=3, layer_sizes=[16, 16, 16, 32], block_attn=true, mid_attn=true,
                                cylindrical=true, compress_Z= true, data_shape=[1, 3, 45, 16, 9], cond_embed= false, time_embed=false)
    cd = CaloDiffusion(torchcd)

    @test cd(data, c, t) ≈ torchcd(torchdata, torchc, torcht) |> fromtorchtensor
end