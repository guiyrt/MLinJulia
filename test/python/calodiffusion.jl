using PyCall, MLinJulia, Test, Underscores

sys = pyimport("sys")
os = pyimport("os")
sys.path = push!(sys.path, "../CaloDiffusion")
pymodels = pyimport("scripts.models")
pyCaloDiffu = pyimport("scripts.CaloDiffu")

@assert get(ENV, "CUDA_VISIBLE_DEVICES", nothing) == ""

@testset "CaloDiffusion noise_pred" begin
    pycalodiffu = pyCaloDiffu.CaloDiffu([1, 45, 16, 9], "../CaloDiffusion/configs/config_dataset2.json")

    c = TrainingConfig("../configs/ds2_electron.yml")
    unet = CondUnet(pycalodiffu.model)

    # Test noise schedule variables
    @test c.sched.α ≈ pycalodiffu.alphas |> fromtorchtensor
    @test c.sched.β ≈ pycalodiffu.betas |> fromtorchtensor
    @test c.sched.sqrt_α⁻¹ ≈ pycalodiffu.sqrt_recip_alphas |> fromtorchtensor
    @test c.sched.sqrt_αcumprod ≈ pycalodiffu.sqrt_alphas_cumprod |> fromtorchtensor
    @test c.sched.sqrt_one_minus_αcumprod ≈ pycalodiffu.sqrt_one_minus_alphas_cumprod |> fromtorchtensor
    @test c.sched.posterior_variance ≈ pycalodiffu.posterior_variance |> fromtorchtensor

    data, torchdata = rand32tensors(9, 16, 45, 1, 32)
    e, torche = rand32tensors(1, 32)
    t, torcht = randinttensors(1:c.nsteps, c.batchsize)
    noise, torchnoise = rand32tensors(size(data)...)

    # Python is zero-based, adjust noise steps
    torcht = torcht - 1

    torchloss = @_ pycalodiffu.compute_loss(torchdata, torche, torchnoise, torcht) |> fromtorchtensor(__; dimsreversed=false)
    loss = batchloss(unet, c, data, e, t, noise)

    @test loss ≈ torchloss[1]
end