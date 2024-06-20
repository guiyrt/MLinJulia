using Flux, TensorCast


struct ConvBlock{T<:Union{Conv, CylindricalConv}}
    conv::T
    norm::GroupNorm
end

function ConvBlock{T}(dim::Int, dim_out::Int, groups::Int=8) where {T<:Union{Conv, CylindricalConv}}
    ConvBlock{T}(
        T((3,3,3), dim=>dim_out; pad=1),
        GroupNorm(dim_out, groups)
    )
end

Flux.@layer ConvBlock

(cb::ConvBlock)(x::AbstractArray) = x |> cb.conv |> cb.norm |> swish


struct ResNetBlock{T<:Union{Conv, CylindricalConv}}
    block₁::ConvBlock{T}
    block₂::ConvBlock{T}
    cond_emb::Union{Dense, Nothing}
    resconv::Union{T, typeof(identity)}
end

function ResNetBlock{T}(dim::Int, dim_out::Int, cond_emb_dim::Union{Int, Nothing}, groups::Int) where {T<:Union{Conv, CylindricalConv}}
    ResNetBlock(
        ConvBlock{T}(dim, dim_out, groups),
        ConvBlock{T}(dim_out, dim_out, groups),
        isnothing(cond_emb_dim) ? nothing : Dense(cond_emb_dim => dim_out) |> fp32,
        dim == dim_out ? nothing : T((1,1,1), dim => dim_out)
    )
end

Flux.@layer ResNetBlock

function (rnb::ResNetBlock)(x::AbstractArray, time::Union{AbstractArray, Nothing}=nothing)
    h = rnb.block₁(x)

    if !isnothing(rnb.cond_emb) && !isnothing(time)
        t = (rnb.cond_emb∘swish)(time)
        h .+= @cast _[1, 1, 1, c, b] := t[c, b]
    end

    rnb.block₂(h) + rnb.resconv(x)
end