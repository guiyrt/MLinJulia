using Flux, PyCall, CUDA
torch = pyimport("torch")


function rand32tensors(size...; togpu::Bool=false)
    data = rand32(size...)
    torchdata = torch.Tensor(data |> reversedims).to(device=(togpu ? "cuda" : "cpu"))

    return data |> fn_if(gpu, togpu), torchdata
end

function randinttensors(options, size...; togpu::Bool=false)
    data = rand(options, size...)
    torchdata = torch.from_numpy(data).to(device=(togpu ? "cuda" : "cpu"))

    return data |> fn_if(gpu, togpu), torchdata
end