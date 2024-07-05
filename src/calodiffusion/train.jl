using ArgParse, Flux, ProgressMeter, Underscores, MLinJulia, Statistics

extract(values::Vector, indices::Vector{Int}) = @_ NNlib.gather(values, indices) |> reshape(__, (1, 1, 1, 1, :))


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

sched = CosineSchedule(c.nsteps)
train, val, test = get_dataloaders(c)

convtype = c.cylindricalconv ? CylindricalConv : Conv
model = CondUnet{convtype}(c.calo.shape[1:3], c.ϕ_input ? 4 : 3, c.blocksize_unet)
optim = Flux.setup(Adam(c.learning_rate), model)

p = Progress(c.epochs * (length(train) + length(val)); showspeed=true)


function applynoise(x::T, noise::T, t::Vector{Int}) where {T<:AbstractArray}
    t[1] == 1 ? x : extract(sched.sqrt_αcumprod, t) .* x + extract(sched.sqrt_one_minus_αcumprod, t) .* noise
end

function noise_weighthed_avg_loss(x::AbstractArray, x̂::AbstractArray, t_emd::AbstractArray, nvoxels::Int)
    weight = 1f0 +  1f0 ./ (t_emd .^ 2f0)
    return sum(weight .* ((x̂ .- x) .^ 2f0)) ./ (mean(weight) * nvoxels)
end

RϕZ_images = createRZϕ_images(c.calo.shape, c.batchsize; ϕ_input=c.ϕ_input)


for epoch in 1:c.epochs
    losses = Float32[]
    val_losses = Float32[]
    val_loss = NaN
    
    for (step, (x, e)) in enumerate(train)
        loss, grads = Flux.withgradient(model) do m
            t = rand(1:c.nsteps, c.batchsize)
            t_emb = @_ t |> NNlib.gather(sched.sqrt_one_minus_αcumprod, __) |> reshape(__, (1, :))

            noise = rand32(size(x)...)
            xₜ = applynoise(x, noise, t)

            x̂ = @_ xₜ |> cat(__, RϕZ_images; dims=4) |> m(__, e, t_emb)

            c.noise_pred_loss ? Flux.mse(noise, x̂) : noise_weighthed_avg_loss(x, x̂, extract(sched.sqrt_one_minus_αcumprod, t), c.calo.nvoxels)
        end
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