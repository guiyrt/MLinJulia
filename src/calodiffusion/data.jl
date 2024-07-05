function preprocess(c::TrainingConfig, showers::AbstractArray, energies::AbstractArray)
    showers ./= 1000f0
    energies ./= 1000f0

    showers ./= energies .* c.maxdeposit

    # Data transformation
    if c.shower_transforms[1] == "sqrt"
        showers = sqrt.(showers)
    elseif c.shower_transforms[1] == "log"
        showers = log.(showers)
        replace!(showers, NaN => min)
    elseif c.shower_transforms[1] == "logit"
        δ = 1f-6
        showers = showers .* (1f0 - 2δ) .+ δ
        showers = log.(showers ./ (1f0 .- showers))
        replace!(showers, NaN => 0f0)
    end

    # Data normalization
    if c.shower_transforms[2] == "norm"
        # Normalize so that x̄=0 and σ²=1
        showers = (showers .- c.s_mean) ./ c.s_std
    elseif c.shower_transforms[2] == "scaled"
        # Scale to range -1 to 1
        showers = c.shower_transforms[1] == "sqrt" ? (showers .* 2f0) .- 1f0 : 2f0 .* (showers .- c.s_min) ./ (c.smax-c.s_min) .- 1f0
    end

    energies = c.e_log ? log10.(energies ./ c.e_min) ./ log10(c.e_max / c.e_min) : (energies .- c.e_min) ./ (c.e_max - c.e_min)

    return reshape(showers, (c.calo.shape..., :)), energies
end


function create_image(shape::NTuple{4, Int}, batchsize::Int, dim::Int)
    step = 1f0 / (shape[dim]-1)
    binvalues = [i*step for i in 0:shape[dim]-1]

    binsize = (shape[1:dim-1]..., 1, shape[dim+1:end]...)
    image = cat([fill(v, binsize) for v in binvalues]...; dims=dim)

    return repeat(image, 1, 1, 1, 1, batchsize)
end


createR_image(shape::NTuple{4, Int}, batchsize::Int) = create_image(shape, batchsize, 1)
createϕ_image(shape::NTuple{4, Int}, batchsize::Int) = create_image(shape, batchsize, 2)
createZ_image(shape::NTuple{4, Int}, batchsize::Int) = create_image(shape, batchsize, 3)


function createRZϕ_images(shape, batchsize; ϕ_input::Bool)
    images = cat(createR_image(shape, batchsize), createZ_image(shape, batchsize); dims=4)

    if ϕ_input
        images = cat(images, createϕ_image(shape, batchsize); dims=4)
    end

    return images
end