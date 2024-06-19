using Flux, NNlib, PyCall


function setparams!(conv::Union{Conv, ConvTranspose}, torchconv::PyObject)
    @assert isclassname(torchconv, ["Conv3d", "ConvTranspose3d"])
    
    params = Dict(k => v for (k, v) in torchconv.named_parameters())
    conv.weight .= params["weight"] |> fromtorchtensor |> NNlib.flipweight
    
    if haskey(params, "bias")
        conv.bias .= params["bias"] |> fromtorchtensor
    end

    return conv
end

function Conv(torchconv::PyObject)
    @assert isclassname(torchconv, ["Conv2d", "Conv3d"])

    conv = Conv(torchconv.kernel_size |> reverse, torchconv.in_channels => torchconv.out_channels;
         stride=torchconv.stride |> reverse, pad=Tuple(x for x in reverse(torchconv.padding) for _ in 1:2),
         dilation=torchconv.dilation |> reverse, groups=torchconv.groups)
    setparams!(conv, torchconv)
end

function ConvTranspose(torchct::PyObject)
    @assert isclassname(torchct, ["ConvTranspose2d", "ConvTranspose3d"])

    ct = ConvTranspose(torchct.kernel_size |> reverse, torchct.in_channels => torchct.out_channels;
         stride=torchct.stride |> reverse, pad=Tuple(x for x in reverse(torchct.padding) for _ in 1:2),
         dilation=torchct.dilation |> reverse)
    setparams!(ct, torchct)
end


function setparams!(gn::GroupNorm, torchgn::PyObject)
    @assert isclassname(torchgn, "GroupNorm")

    params = Dict(k => v for (k, v) in torchgn.named_parameters())
    gn.γ .= params["weight"] |> fromtorchtensor
    gn.β .= params["bias"] |> fromtorchtensor

    return gn
end

function GroupNorm(torchgn::PyObject)
    @assert isclassname(torchgn, "GroupNorm")

    gn = GroupNorm(torchgn.num_channels, torchgn.num_groups; affine=torchgn.affine, eps=Float32(torchgn.eps))
    setparams!(gn, torchgn)
end


function setparams!(c::Chain, torchseq::PyObject)
    @assert isclassname(torchseq, "Sequential")

    for (layer, torchlayer) in zip(c, torchseq)
        setparams!(layer, torchlayer)
    end

    return c
end

function Chain(torchseq::PyObject)
    @assert isclassname(torchseq, "Sequential")

    Chain([frompyclass(pyo) for pyo in torchseq]...)
end


"""
As PyObject is not parametric, multiple dispatch is not possible. Instead, the Python classname has to be checked.
A cool alternative would be to create a parametric PyObject struct, as such:

trygetproperty(x, s::Symbol, default) = hasproperty(x, s) ? getproperty(x, s) : default
struct PyObjectClass{Symbol}
    pyo::PyObject

    PyObjectClass(pyo::PyObject) = new{Symbol(trygetproperty(pyo, :__module__, "builtins"), '.', pyo.__class__.__name__)}(pyo)
end

Like this, a PyTorch Conv2d would have type PyObjectClass{Symbol(torch.nn.modules.conv.Conv2d)} insted of just PyObject.

Then, instead of having a new contructor Conv(torchconv::PyObject), convert could be implemented with
 signature convert(Conv, PyObjectClass{Symbol(torch.nn.modules.conv.Conv2d)}), or a new function fromtorch with methods for 
 every Python class of interest. 

fromtorch makes more sense, fromtorch(pyo::PyObject) could convert PyObject to PyObjectClass and do a recursive call, on which the
 implemented parametric type would take over. Maybe if there is time!

"""
function frompyclass(pyo::PyObject)
    if isclassname(pyo, ["Conv2d", "Conv3d"])
        Conv(pyo)
    elseif isclassname(pyo, ["ConvTranspose2d", "ConvTranspose2d"])
        ConvTranspose(pyo)
    elseif isclassname(pyo, "CylindricalConv")
        CylindricalConv(pyo)
    elseif isclassname(pyo, "CylindricalConvTranspose")
        CylindricalConvTranspose(pyo)
    elseif isclassname(pyo, "GroupNorm")
        GroupNorm(pyo)
    elseif isclassname(pyo, "LinearAttention")
        LinearAttention(pyo)
    elseif isclassname(pyo, "PreNorm")
        PreNorm(pyo)
    elseif isclassname(pyo, "Residual")
        Residual(pyo)
    else
        throw(DomainError("Unsupported layer type: $(pyo.__class__.__name__)"))
    end
end