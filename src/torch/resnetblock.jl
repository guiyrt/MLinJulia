using Flux, PyCall


function setparams!(cb::ConvBlock, torchcb::PyObject)
    @assert isclassname(torchcb, "ConvBlock")

    setparams!(cb.conv, torchcb.conv)
    setparams!(cb.norm, torchcb.norm)

    return cb
end

function ConvBlock(torchcb::PyObject)
    @assert isclassname(torchcb, "ConvBlock")
    convtype = isclassname(torchcb.conv, "CylindricalConv") ? CylindricalConv : Conv
    
    ConvBlock{convtype}(convtype(torchcb.conv), GroupNorm(torchcb.norm))
end


function setparams!(rnb::ResNetBlock, torchrnb::PyObject)
    @assert isclassname(torchrnb, "ResNetBlock")

    setparams!(rnb.block₁, torchrnb.block1)
    setparams!(rnb.block₂, torchrnb.block2)
    
    if !isnothing(rnb.cond_emb)
        setparams!(rnb.cond_emb, torchrnb.cond_emb)
    end

    if rnb.resconv !== identity
        setparams!(rnb.resconv, torchrnb.res_conv)
    end

    return cb
end

function ResNetBlock(torchrnb::PyObject) 
    @assert isclassname(torchrnb, "ResNetBlock")
    convtype = isclassname(torchrnb.block1.conv, "CylindricalConv") ? CylindricalConv : Conv

    ResNetBlock{convtype}(
        ConvBlock(torchrnb.block1),
        ConvBlock(torchrnb.block2),
        isnothing(torchrnb.cond_emb) ? nothing : Dense(get(torchrnb.cond_emb, 1)),
        isclassname(torchrnb.res_conv, "Identity") ? identity : convtype(torchrnb.res_conv)
   )
end