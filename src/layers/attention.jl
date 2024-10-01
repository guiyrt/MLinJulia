using Flux, TensorCast


struct LinearAttention{T<:Union{Conv, CylindricalConv}}
    scale::Float32
    nHeads::Int
    toQkv::T
    toOut::Chain{<:Tuple{T, GroupNorm}}
end

function LinearAttention{T}(dim::Int, nHeads::Int=1, dimHead::Int=32) where {T<:Union{Conv, CylindricalConv}}
    hiddenDim = dimHead*nHeads

    LinearAttention{T}(
        Float32(dimHead^-0.5),
        nHeads,
        T((1,1,1), dim => hiddenDim * 3; bias=false),
        Chain(T((1,1,1), hiddenDim => dim), GroupNorm(dim, 1))
    )
end

Flux.@layer LinearAttention

function (la::LinearAttention)(x::AbstractArray)
    w, h, l, _, _ = size(x)
    
    qkv = la.toQkv(x)
    q = qkv[:, :, :, 1:32, :]
    k = qkv[:, :, :, 33:64, :]
    v = qkv[:, :, :, 65:96, :]
    
    q, k, v = map((t -> @cast _[z⊗y⊗x, c, h, b] := t[z, y, x, c⊗h, b] h in 1:la.nHeads), [q, k, v])

    qₛ = softmax(q; dims=2) .* la.scale
    kₛ = softmax(k; dims=1)

    vt = permutedims(v, (2, 1, 3, 4))
    ct = batched_mul(vt, kₛ)
    c = permutedims(ct, (2, 1, 3, 4))

    out = batched_mul(qₛ, c)

    la.toOut(@cast _[z, y, x, c⊗h, b] := out[z⊗y⊗x, c, h, b] h in 1:la.nHeads, z in 1:w, y in 1:h, x in 1:l)
end


struct PreNorm
    norm::GroupNorm
    layer
end

PreNorm(dim::Int, layer) = PreNorm(GroupNorm(dim, 1), layer)

Flux.@layer PreNorm

(pn::PreNorm)(x::AbstractArray) =  x |> pn.norm |> pn.layer


struct Residual
    layer
end

Flux.@layer Residual

(res::Residual)(x::AbstractArray) = res.layer(x) + x