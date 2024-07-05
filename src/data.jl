using HDF5, YAML, Flux, Underscores


lastdimcat(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N} = cat(x, y, ndims(x))
lastdimcat(x::Tuple{T,N}, y::Tuple{T,N}) where {T,N} = (lastdimcat(x[1], y[1]), lastdimcat(x[2], y[2]))

load_datafile(filename::String) = h5read(filename, "showers") |> f32, h5read(filename, "incident_energies") |> f32
load_dataset(filenames::Vector{String}) = reduce(lastdimcat, [load_datafile(file) for file in filenames])

function get_dataloaders(c::TrainingConfig)
    train = @_ load_dataset(c.trainfiles) |> preprocess(c, __...)
    test = @_ load_dataset(c.testfiles) |> preprocess(c, __...)

    train, val = Flux.splitobs(train, at=c.train_val_split, shuffle=true)

    train_loader = Flux.DataLoader(train, batchsize=c.batchsize, shuffle=true)
    val_loader = Flux.DataLoader(val, batchsize=c.batchsize, shuffle=false)
    test_loader = Flux.DataLoader(test, batchsize=c.batchsize, shuffle=false)

    return train_loader, val_loader, test_loader
end