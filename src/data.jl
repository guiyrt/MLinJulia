using HDF5, XML, IterTools, YAML, Flux


function loadDataFile(filename::String, c::TrainingConfig)
    showers::Array{Float32} = h5read("datasets/$filename", "showers")
    ienergies::Array{Float32} = h5read("datasets/$filename", "incident_energies")

    showers ./= 1000.
    ienergies ./= 1000.

    showers ./= ienergies.*c.maxDeposit

    # Data transformation
    if c.showerTransforms[1] == "sqrt"
        showers = sqrt.(showers)
    elseif c.showerTransforms[1] == "log"
        showers = log.(showers)
        replace!(showers, NaN => min)
    elseif c.showerTransforms[1] == "logit"
        δ = 1e-6
        showers = showers .* (1-2δ) .+ δ
        showers = log.(showers ./ (1 .- showers))
        replace!(showers, NaN => 0.)
    end

    # Data normalization
    if c.showerTransforms[2] == "norm"
        # Normalize so that x̄=0 and σ²=1
        showers = (showers .- c.sMean) ./ c.sStd
    elseif c.showerTransforms[2] == "scaled"
        # Scale to range -1 to 1
        showers = c.showerTransforms[1] == "sqrt" ? (showers.*2.0) .- 1.0 : 2.0 .* (showers.-c.sMin) ./ (c.sMax-c.sMin) .- 1.0
    end

    ienergies = c.logE ? log10.(ienergies./c.eMin) ./ log10(c.eMax/c.eMin) : (ienergies.-c.eMin) ./ (c.eMax-c.eMin)

    showers = reshape(showers, (c.showerShape..., :))
    ienergies = reshape(ienergies, (:))

    return showers, ienergies
end

tupleLastDimCat(x::Tuple{T,N}, y::Tuple{T,N}) where {T,N} = (cat(x[1], y[1], dims=(length∘size)(x[1])), cat(x[2], y[2], dims=(length∘size)(x[2])))

loadDataset(filenames::Vector{String}, c::TrainingConfig) = reduce(tupleLastDimCat, [loadDataFile(trainFile, c) for trainFile in filenames])

function getDataLoaders(c::TrainingConfig)
    train, val = Flux.splitobs(loadDataset(c.trainFiles, c), at=c.trainValSplit, shuffle=true)
    eval = loadDataset(c.evalFiles, c)

    trainLoader = Flux.DataLoader(train, batchsize=c.batchSize, shuffle=true)
    valLoader = Flux.DataLoader(val, batchsize=c.batchSize, shuffle=false)
    evalLoader = Flux.DataLoader(eval, batchsize=c.batchSize, shuffle=false)

    return trainLoader, valLoader, evalLoader
end