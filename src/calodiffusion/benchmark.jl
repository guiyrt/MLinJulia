using ArgParse, Flux, MLinJulia, BenchmarkTools, CUDA


function parse_commandline()
    s = ArgParseSettings(prog="Profile training step with BenchmarkTools")

    @add_arg_table s begin
        "config"
            help = "Path to YAML config file"
            arg_type = String
        
        "--batchsize", "-b"
            help = "Number of samples per batch"
            arg_type = Int

        "--device", "-d"
            help = "Type of device (cpu/gpu) to use for benchmark"
            arg_type = String

        "--steps", "-s"
            help = "Train steps sampled for benchmarking"
            arg_type = Int
            default = 20
        
        "--warmup", "-w"
            help = "Warm-up steps before profiling"
            arg_type = Int
            default = 5

        "--output", "-o"
            help = "Path (without extension) to save ouput files"
            arg_type = String
            default = "$(pwd())/benchmark"
    end

    return parse_args(s)
end

args = parse_commandline()
c = TrainingConfig(args["config"], args["batchsize"], args["device"])

train, _, _ = get_dataloaders(c)
model = CondUnet{c.convtype}(c.calo.shape[1:3], c.nchannels, c.blocksize_unet) |> c.device
optim = Flux.setup(Adam(c.learning_rate), model)

function step(model, dataloader)
    data, _ = iterate(dataloader)
    (x, e) = data
    t = rand(1:c.nsteps, size(x)[end]) |> c.device
    noise = c.device == gpu ? CUDA.rand(size(x)...) : rand32(size(x)...)
    _, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
    Flux.update!(optim, model, grads[1])
end

# Warm-up
for _ in 1:args["warmup"]
    step(model, train)
end

# Benchmark
if c.device == cpu
    results = @benchmark step(model, train) samples=args["steps"] evals=1 seconds=3600
else
    results = @benchmark begin
        CUDA.@sync step(model, train)
    end samples=args["steps"] evals=1 seconds=3600
end

# Save JSON
BenchmarkTools.save("$(args["output"]).json", results)

# Save summary
open("$(args["output"]).txt","w") do f
    if c.device == gpu
        redirect_stdout(f) do
            CUDA.@time step(model, train)
        end
    end

    show(f, "text/plain", results)
end