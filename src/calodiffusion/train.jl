using ArgParse, Flux, ProgressMeter, Underscores, MLinJulia, Statistics, CUDA, Printf, Dates, JLD2


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
min_val_loss = Inf

# Create run folder and copy config
runpath = mkpath("runs/$(Dates.now())")
cp(args["config"], joinpath(runpath, basename(args["config"])))

for epoch in 1:c.epochs
    # Training loop
    p = Progress(length(train); showspeed=true)
    losses = Float32[]
    for (step, (x, e)) in enumerate(train)
        t = rand(1:c.nsteps, size(x)[end]) |> c.device
        noise = c.device == gpu ? CUDA.rand(size(x)...) : rand32(size(x)...)
        loss, grads = Flux.withgradient(batchloss, model, c, x, e, t, noise)
        Flux.update!(optim, model, grads[1])

        push!(losses, loss)
        next!(p, showvalues=[(:epoch, epoch), (:step, step), (:loss, mean(losses))])
        break
    end

    # Validation loop
    val_p = Progress(length(val); showspeed=true)
    val_losses = Float32[]
    for (step, (x, e)) in enumerate(val)
        t = rand(1:c.nsteps, size(x)[end]) |> c.device
        noise = c.device == gpu ? CUDA.rand(size(x)...) : rand32(size(x)...)
        val_loss = batchloss(model, c, x, e, t, noise)

        push!(val_losses, val_loss)
        next!(val_p, showvalues=[(:epoch, epoch), (:step, step), (:val_loss, mean(val_losses))])
    end

    # Save model with lowest mean val_loss
    if mean(val_losses) < min_val_loss
        jldsave("$runpath/min_val_loss.jld2", model_state = Flux.state(model |> cpu))
        global min_val_loss = mean(val_losses)
    end
end

# Test loop
test_p = Progress(length(train); showspeed=true)
test_losses = Float32[]
for (step, (x, e)) in enumerate(test)
    t = rand(1:c.nsteps, size(x)[end]) |> c.device
    noise = c.device == gpu ? CUDA.rand(size(x)...) : rand32(size(x)...)
    test_loss = batchloss(model, c, x, e, t, noise)

    push!(test_losses, test_loss)
    next!(test_p, showvalues=[(:step, step), (:test_loss, mean(test_losses))])
end

# Save final model
jldsave(@sprintf("%s/final_%.2f.jld2", runpath, mean(test_losses)), model_state = Flux.state(model |> cpu))