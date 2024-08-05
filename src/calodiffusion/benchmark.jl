using Flux, MLinJulia, Statistics, CUDA, Printf, ArgParse, Underscores

function parse_commandline()
    s = ArgParseSettings(prog="Benchmark CaloDiffusion training loop")

    @add_arg_table s begin
        "config"
            help = "Path to YAML config file"
            arg_type = String
            default = "configs/ds2_electron.yml"

        "--steps", "-s"
            help = "Train steps for benchmark"
            arg_type = Int
            default = 100
    end

    return parse_args(s)
end

function gpu_mem_info(used_gpumem::Union{Nothing,Int64}=nothing, info::CUDA.MemoryInfo=CUDA.MemoryInfo())
    used_bytes = isnothing(used_gpumem) ? info.total_bytes - info.free_bytes : used_gpumem
    used_ratio = used_bytes / info.total_bytes
    return used_bytes, @sprintf("GPU memory: %s (%.2f%%)", Base.format_bytes(used_bytes), 100*used_ratio)
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

gpumemₛ = Int64[]

# Benchmark
for s in 1:args["steps"]
    startₜ = time()
    
    # Get mini-batch
    t₁ = time()
    data, _ = iterate(train)
    (x, e) = data
    t₂ = time()
    push!(minibatchₜ, (t₂-t₁)*1000)

    # Generate random noise time steps
    t₁ = time()
    t = rand(1:c.nsteps, size(x)[end]) |> c.device
    t₂ = time()
    push!(tₜ, (t₂-t₁)*1000)
    
    # Generate Gaussian noise
    t₁ = time()
    noise = rand32(size(x)...) |> gpu
    t₂ = time()
    push!(noiseₜ, (t₂-t₁)*1000)
    
    # Forward pass and calculate loss
    t₁ = time()
    _, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
    t₂ = time()
    push!(forwardₜ, (t₂-t₁)*1000)
    
    # Update weights
    t₁ = time()
    Flux.update!(optim, model, grads[1])
    t₂ = time()
    push!(backwardₜ, (t₂-t₁)*1000)

    # Track total training step time
    endₜ = time()
    push!(totalₜ, (endₜ-startₜ)*1000)

    # Check GPU memory usage
    gpu_usedmem, gpu_info_str = gpu_mem_info()
    push!(gpumemₛ, gpu_usedmem)
    
    # Print single step time
    @printf(
        "[%03i] %s | Mini-batch: %.3fms | Time steps: %.3fms | Noise: %.3fms | Forward-pass: %.3fms | Backward-pass: %.3fms | TOTAL: %.3fms\n",
        s, gpu_info_str, minibatchₜ[s], tₜ[s], noiseₜ[s], forwardₜ[s], backwardₜ[s], totalₜ[s]
    )
end

# Separate final benchmark results
println("-"^168)

# Benchamrk mean, with first execution subtracted
_, avg_gpu_info_str = ceil(Int64, mean(gpumemₛ)) |> gpu_mem_info
@printf(
    "[Avg] %s | Mini-batch: %.3fms | Time steps: %.3fms | Noise: %.3fms | Forward-pass: %.3fms | Backward-pass: %.3fms | TOTAL: %.3fms\n",
    avg_gpu_info_str, mean(minibatchₜ[2:end]), mean(tₜ[2:end]), mean(noiseₜ[2:end]), mean(forwardₜ[2:end]), mean(backwardₜ[2:end]), mean(totalₜ[2:end])
)

# Benchmark max 
_, max_gpu_info_str = gpumemₛ |> maximum |> gpu_mem_info
@printf(
    "[Max] %s | Mini-batch: %.3fms | Time steps: %.3fms | Noise: %.3fms | Forward-pass: %.3fms | Backward-pass: %.3fms | TOTAL: %.3fms\n",
    max_gpu_info_str, maximum(minibatchₜ[2:end]), maximum(tₜ[2:end]), maximum(noiseₜ[2:end]),
    maximum(forwardₜ[2:end]), maximum(backwardₜ[2:end]), maximum(totalₜ[2:end])
)