using ArgParse, Flux, MLinJulia, CUDA, NVTX


function parse_commandline()
    s = ArgParseSettings(prog="Train CaloDiffusion")

    @add_arg_table s begin
        "config"
            help = "Path to YAML config file"
            arg_type = String
            required = true

        "--steps", "-s"
            help = "Train steps for benchmark"
            arg_type = Int
            default = 100
    end

    return parse_args(s)
end

args = parse_commandline()
c = TrainingConfig(args["config"])

train, _, _ = get_dataloaders(c, c.device)

model = CondUnet{c.convtype}(c.calo.shape[1:3], c.nchannels, c.blocksize_unet) |> c.device
optim = Flux.setup(Adam(c.learning_rate), model)

# Training loop
CUDA.@profile begin
    for s in 1:args["steps"]
        NVTX.@range "step" begin
            NVTX.@range "Data from dataloader" begin
                data, _ = iterate(train)
                (x, e) = data
            end

            NVTX.@range "Random t" begin
                t = rand(1:c.nsteps, size(x)[end]) |> c.device
            end

            NVTX.@range "Random noise" begin
                noise = c.device == gpu ? CUDA.rand(size(x)...) : rand32(size(x)...)
            end
            
            NVTX.@range "Get gradients from loss" begin
                loss, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
            end

            NVTX.@range "Update weights" begin
                Flux.update!(optim, model, grads[1])
            end
        end
    end
end
