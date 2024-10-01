using PyCall, MLinJulia, Test, Underscores

sys = pyimport("sys")
os = pyimport("os")
sys.path = push!(sys.path, "../CaloDiffusion")
pymodels = pyimport("scripts.models")
pyCaloDiffu = pyimport("scripts.CaloDiffu")

@testset "CaloDiffusion noise_pred_loss" begin
    pycalodiffu = pyCaloDiffu.CaloDiffu([1, 45, 16, 9], "../CaloDiffusion/configs/config_dataset2.json")

    c = TrainingConfig("configs/ds2_electron.yml")
    unet = CondUnet(pycalodiffu.model) |> c.device

    # Test noise schedule variables
    @test c.sched.α |> cpu ≈ pycalodiffu.alphas |> fromtorchtensor
    @test c.sched.β |> cpu≈ pycalodiffu.betas |> fromtorchtensor
    @test c.sched.sqrt_α⁻¹|> cpu ≈ pycalodiffu.sqrt_recip_alphas |> fromtorchtensor
    @test c.sched.sqrt_αcumprod|> cpu ≈ pycalodiffu.sqrt_alphas_cumprod |> fromtorchtensor
    @test c.sched.sqrt_one_minus_αcumprod|> cpu ≈ pycalodiffu.sqrt_one_minus_alphas_cumprod |> fromtorchtensor
    @test c.sched.posterior_variance|> cpu ≈ pycalodiffu.posterior_variance |> fromtorchtensor

    data, torchdata = rand32tensors(9, 16, 45, 1, c.batchsize; togpu=CUDA.functional())
    e, torche = rand32tensors(1, 32; togpu=CUDA.functional())
    t, torcht = randinttensors(1:c.nsteps, c.batchsize; togpu=CUDA.functional())
    noise, torchnoise = rand32tensors(size(data)...; togpu=CUDA.functional())

    # Python is zero-based, adjust noise steps
    torcht = torcht - 1

    torchloss = @_ pycalodiffu.compute_loss(torchdata, torche, torchnoise, torcht) |> fromtorchtensor(__; dimsreversed=false)[1]
    loss = batchloss(unet, c, data, e, t, noise)

    @test loss ≈ torchloss
end

@testset "CaloDiffusion noise_weighthed_avg_loss" begin
    pycalodiffu = pyCaloDiffu.CaloDiffu([1, 45, 50, 18], "../CaloDiffusion/configs/config_dataset3.json")

    c = TrainingConfig("configs/ds3_electron.yml")
    unet = CondUnet(pycalodiffu.model) |> c.device

    data, torchdata = rand32tensors(18, 50, 45, 1, c.batchsize; togpu=CUDA.functional())
    e, torche = rand32tensors(1, c.batchsize; togpu=CUDA.functional())
    t, torcht = randinttensors(1:c.nsteps, c.batchsize; togpu=CUDA.functional())
    noise, torchnoise = rand32tensors(size(data)...; togpu=CUDA.functional())

    # Python is zero-based, adjust noise steps
    torcht = torcht - 1

    torchloss = @_ pycalodiffu.compute_loss(torchdata, torche, torchnoise, torcht) |> fromtorchtensor(__; dimsreversed=false)[1]
    loss = batchloss(unet, c, data, e, t, noise)

    @test loss ≈ torchloss
end