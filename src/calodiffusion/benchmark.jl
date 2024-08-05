using Flux, MLinJulia, Statistics, CUDA, Printf, ArgParse

function parse_commandline()
    s = ArgParseSettings(prog="Benchmark CaloDiffusion training loop")

    @add_arg_table s begin
        "config"
            help = "Path to YAML config file"
            arg_type = String
            default = "configs/ds2_electron.yml"

        "steps"
            help = "Train steps for benchmark"
            arg_type = Int
            default = 100
    end

    return parse_args(s)
end

# Build dataloaders model from yml config file
args = parse_commandline()
c = TrainingConfig(args["config"])
train, _, _ = get_dataloaders(c, c.device)
model = CondUnet{c.convtype}(c.calo.shape[1:3], c.nchannels, c.blocksize_unet) |> c.device
optim = Flux.setup(Adam(c.learning_rate), model)

# Keep track of execution times
minibatchₜ = Float64[]
tₜ = Float64[]
noiseₜ = Float64[]
forwardₜ = Float64[]
backwardₜ = Float64[]
totalₜ = Float64[]

# Benchmark
for s in 1:args["steps"]
    # Get mini-batch
    t0 = time()
    data, _ = iterate(train)
    (x, e) = data
    t1 = time()
    push!(minibatchₜ, (t1-t0)*1000)

    # Generate random noise time steps
    t0 = time()
    t = rand(1:c.nsteps, size(x)[end]) |> c.device
    t1 = time()
    push!(tₜ, (t1-t0)*1000)
    
    # Generate Gaussian noise
    t0 = time()
    noise = rand32(size(x)...) |> c.device
    t1 = time()
    push!(noiseₜ, (t1-t0)*1000)
    
    # Forward pass and calculate loss
    t0 = time()
    loss, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
    t1 = time()
    push!(forwardₜ, (t1-t0)*1000)
    
    # Update weights
    t0 = time()
    Flux.update!(optim, model, grads[1])
    t1 = time()
    push!(backwardₜ, (t1-t0)*1000)
    
    # Print single step time
    push!(totalₜ, minibatchₜ[s] + tₜ[s] + noiseₜ[s] + forwardₜ[s] + backwardₜ[s])
    @printf "[%03i] Mini-batch: %.3fms | Time steps: %.3fms | Noise: %.3fms | Forward-pass: %.3fms | Backward-pass: %.3fms | TOTAL: %.3fms\n" s minibatchₜ[s] tₜ[s] noiseₜ[s] forwardₜ[s] backwardₜ[s] totalₜ[s]
end

# Mean execution times, with first execution subtracted
println("-"^134)
@printf "[Avg] Mini-batch: %.3fms | Time steps: %.3fms | Noise: %.3fms | Forward-pass: %.3fms | Backward-pass: %.3fms | TOTAL: %.3fms\n" mean(minibatchₜ[2:end]) mean(tₜ[2:end]) mean(noiseₜ[2:end]) mean(forwardₜ[2:end]) mean(backwardₜ[2:end]) mean(totalₜ[2:end])