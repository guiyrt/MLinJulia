using Flux


struct CylindricalConv
    circPad::NTuple{6, Int64}
    conv::Conv
end

function CylindricalConv(k::NTuple{3,Integer}, ch::Pair{<:Integer,<:Integer}, σ=identity;
    init=Flux.glorot_uniform, stride=1, pad=0, dilation=1, groups=1, bias=true
)
    CylindricalConv(
        (0, 0, pad, pad, 0, 0),
        Conv(k, ch, σ; init, stride, pad=(pad, pad, 0, 0, pad, pad), dilation, groups, bias)
    )
end

Flux.@layer CylindricalConv

(c::CylindricalConv)(x::AbstractArray) = pad_circular(x, c.circPad) |> c.conv


struct CylindricalConvTranspose
    circPad::NTuple{6, Int64}
    convTranspose::ConvTranspose
end

function CylindricalConvTranspose(k::NTuple{3,Integer}, ch::Pair{<:Integer,<:Integer}, σ=identity;
    init=Flux.glorot_uniform, stride=(2, 1, 1), pad=1, outpad=0, dilation=1, groups=1
)
    CylindricalConvTranspose(
        (0, 0, pad, pad, 0, 0),
        ConvTranspose(k, ch, σ; init, stride, pad=(pad, pad, k[2]-1, k[2]-1, pad, pad), outpad, dilation, groups)
    )
end

Flux.@layer CylindricalConvTranspose

(c::CylindricalConvTranspose)(x::AbstractArray) = pad_circular(x, c.circPad) |> c.convTranspose