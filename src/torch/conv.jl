using Flux, PyCall


function setparams!(cc::CylindricalConv, torchcc::PyObject)
    @assert isclassname(torchcc, "CylindricalConv")

    setparams!(cc.conv, torchcc.conv)
    return cc
end

function CylindricalConv(torchcc::PyObject)
    @assert isclassname(torchcc, "CylindricalConv")
    
    CylindricalConv(torchcc.circ_pad |> reverse, Conv(torchcc.conv))
end


function setparams!(cct::CylindricalConvTranspose, torchcct::PyObject)
    @assert isclassname(torchcct, "CylindricalConvTranspose")

    setparams!(cct.convTranspose, torchcct.conv_transpose)
    return cct
end

function CylindricalConvTranspose(torchcct::PyObject)
    @assert isclassname(torchcct, "CylindricalConvTranspose")
    
    CylindricalConvTranspose(torchcct.circ_pad |> reverse, ConvTranspose(torchcct.conv_transpose))
end
