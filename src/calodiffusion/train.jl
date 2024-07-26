using ArgParse, Flux, ProgressMeter, Underscores, MLinJulia, Statistics


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

train, val, test = get_dataloaders(c)

model = CondUnet{c.convtype}(c.calo.shape[1:3], c.nchannels, c.blocksize_unet)
optim = Flux.setup(Adam(c.learning_rate), model)

p = Progress(c.epochs * (length(train) + length(val)); showspeed=true)


for epoch in 1:c.epochs
    losses = Float32[]
    val_losses = Float32[]
    val_loss = NaN
    
    for (step, (x, e)) in enumerate(train)
        t = rand(1:c.nsteps, c.batchsize)
        noise = rand32(size(x)...)
        loss, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
        Flux.update!(optim, model, grads[1])

        push!(losses, loss)
        next!(p, showvalues=[(:epoch, epoch), (:step, step), (:loss, mean(losses)), (:val_loss, mean(val_losses))])
    end

    # for (step, (shower, energy)) in enumerate(val)
    #     time = rand(1:c.nsteps, c.batchsize)
    #     val_loss = diffusionloss(model, shower, energy, time; noise_pred_loss=c.noise_pred_loss)

    #     push!(val_losses, val_loss)
    #     next!(p, showvalues=[(:epoch, epoch), (:step,step), (:loss,mean(losses)), (:val_acc, val_acc)])
    # end
end