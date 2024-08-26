using ArgParse, Flux, MLinJulia, CUDA, NVTX

NVTX.enable_gc_hooks(;gc=true, alloc=true, free=true)
NVTX.enable_inference_hook(true)

function parse_commandline()
    s = ArgParseSettings(prog="Profile training step with external CUDA profiler")

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
            help = "Train steps for profile"
            arg_type = Int
            default = 20

        "--warmup", "-w"
            help = "Warm-up steps before profiling"
            arg_type = Int
            default = 5
    end

    return parse_args(s)
end

args = parse_commandline()
c = TrainingConfig(args["config"], args["batchsize"], args["device"])

train, _, _ = get_dataloaders(c)
model = CondUnet{c.convtype}(c.calo.shape[1:3], c.nchannels, c.blocksize_unet) |> c.device
optim = Flux.setup(Adam(c.learning_rate), model)

function step(model, dataloader)
    NVTX.@range "Step" begin
        NVTX.@range "Get batch" begin
            data, _ = iterate(dataloader)
            (x, e) = data
        end

        NVTX.@range "Random t" begin
            t = rand(1:c.nsteps, size(x)[end]) |> c.device
        end

        NVTX.@range "Random noise" begin
            noise = c.device == gpu ? CUDA.rand(size(x)...) : rand32(size(x)...)
        end
        
        NVTX.@range "Forward/backward passes" begin
            _, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
        end

        NVTX.@range "Update weights" begin
            Flux.update!(optim, model, grads[1])
        end
    end
end

# Warm-up
for _ in 1:args["warmup"]
    step(model, train)
end

# Training loop
CUDA.@profile external=true begin
    for s in 1:args["steps"]
        step(model, train)     
    end
end
