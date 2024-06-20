using Flux, PyCall


function setparams!(la::LinearAttention, torchla::PyObject)
    @assert isclassname(torchla, "LinearAttention")

    setparams!(la.toQkv, torchla.to_qkv)
    setparams!(la.toOut, torchla.to_out)
    return la
end

function LinearAttention(torchla::PyObject)
    @assert isclassname(torchla, "LinearAttention")
    convtype = isclassname(torchla.to_qkv, "CylindricalConv") ? CylindricalConv : Conv

    LinearAttention{convtype}(Float32(torchla.scale), torchla.n_heads, convtype(torchla.to_qkv), Chain(torchla.to_out))
end


function setparams!(pn::PreNorm, torchpn::PyObject)
    @assert isclassname(torchpn, "PreNorm")
    
    setparams!(pn.norm, torchpn.norm)
    setparams!(pn.layer, torchpn.layer)
end

function PreNorm(torchpn::PyObject)
    @assert isclassname(torchpn, "PreNorm")

    PreNorm(GroupNorm(torchpn.norm), frompyclass(torchpn.layer))
end


function setparams!(res::Residual, torchres::PyObject)
    @assert isclassname(torchres, "Residual")
    
    setparams!(res.layer, torchres.layer)
end

function Residual(torchres::PyObject)
    @assert isclassname(torchres, "Residual")

    Residual(frompyclass(torchres.layer))
end