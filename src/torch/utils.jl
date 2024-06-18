using Flux, CUDA, PyCall

fromtorchtensor(t::PyObject; togpu::Bool=false) = (t.device.type == "cuda" ? t.cpu() : t).detach().numpy() |> reversedims |> fn_if(gpu, togpu)

totorchtensor(t::AbstractArray; togpu::Bool=false) = torch.Tensor(t |> fn_if(cpu, t isa CuArray) |> reversedims).to(device=(togpu ? "cuda" : "cpu"))

isclassname(pyo::PyObject, classname::String) = pyo.__class__.__name__ == classname
isclassname(pyo::PyObject, classnames::Vector{String}) = any(isclassname(pyo, classname) for classname in classnames)