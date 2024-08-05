using ArgParse, Flux, ProgressMeter, Underscores, MLinJulia, Statistics, CUDA


function parse_commandline()
    s = ArgParseSettings(prog="Train CaloDiffusion")

    @add_arg_table s begin
        "config"
            help = "Path to YAML config file"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

args = parse_commandline()
c = TrainingConfig(args["config"])

train, val, test = get_dataloaders(c, c.device)

model = CondUnet{c.convtype}(c.calo.shape[1:3], c.nchannels, c.blocksize_unet) |> c.device
optim = Flux.setup(Adam(c.learning_rate), model)

for epoch in 1:c.epochs
    p = Progress(length(train); showspeed=true)
    losses = Float32[]
    for (step, (x, e)) in enumerate(train)
        t = rand(1:c.nsteps, size(x)[end]) |> c.device
        noise = rand32(size(x)...) |> c.device
        loss, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
        Flux.update!(optim, model, grads[1])

        push!(losses, loss)
        next!(p, showvalues=[(:epoch, epoch), (:step, step), (:loss, mean(losses))])
    end

    val_p = Progress(length(val); showspeed=true)
    val_losses = Float32[]
    for (step, (x, e)) in enumerate(val)
        t = rand(1:c.nsteps, size(x)[end]) |> c.device
        noise = rand32(size(x)...) |> c.device
        val_loss = batchloss(model, c, x, e, t, noise)

        push!(val_losses, val_loss)
        next!(val_p, showvalues=[(:epoch, epoch), (:step,step), (:val_losses,mean(val_losses))])
    end
end