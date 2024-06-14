using Flux


struct CylindricalConvTranspose
    circPad::NTuple{6, Int64}
    convTranspose::ConvTranspose
end

function CylindricalConvTranspose(k::NTuple{3,Integer}, ch::Pair{<:Integer,<:Integer}, σ=identity;
    init=Flux.glorot_uniform, stride=2, pad::Int=1, dilation=1, groups=1, bias=true
)
    convTransposePad = (pad, pad, k[2]-1, k[2]-1, pad, pad)
    CylindricalConvTranspose(
        (0, 0, pad, pad, 0, 0),
        ConvTranspose(k, ch, σ; init, stride, pad=convTransposePad, dilation, groups, bias)
    )
end

Flux.@layer CylindricalConvTranspose

(c::CylindricalConvTranspose)(x::AbstractArray) = pad_circular(x, c.circPad) |> c.convTranspose


struct CylindricalConv
    circPad::NTuple{6, Int64}
    conv::Conv
end

function CylindricalConv(k::NTuple{3,Integer}, ch::Pair{<:Integer,<:Integer}, σ=identity;
    init=Flux.glorot_uniform, stride=2, pad::Int=1, dilation=1, groups=1, bias=true
)
    convPad = (pad, pad, 0, 0, pad, pad)
    CylindricalConv(
        (0, 0, pad, pad, 0, 0),
        Conv(k, ch, σ; init, stride, pad=convPad, dilation, groups, bias)
    )
end

Flux.@layer CylindricalConv

(c::CylindricalConv)(x::AbstractArray) = pad_circular(x, c.circPad) |> c.conv