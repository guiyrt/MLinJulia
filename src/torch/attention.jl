using Flux, PyCall


function setparams!(la::LinearAttention, torchla::PyObject)
    @assert isclassname(torchla, "LinearAttention")

    setparams!(la.toQkv, torchla.to_qkv)
    setparams!(la.toOut, torchla.to_out)
    return la
end

function LinearAttention(torchla::PyObject)
    @assert isclassname(torchla, "LinearAttention")
    convType = isclassname(torchla.to_qkv, "CylindricalConv") ? CylindricalConv : Conv

    LinearAttention{convType}(Float32(torchla.scale), torchla.n_heads, convType(torchla.to_qkv), Chain(torchla.to_out))
end