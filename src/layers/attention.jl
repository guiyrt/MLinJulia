using Flux, TensorCast


struct LinearAttention{T<:Union{Conv, CylindricalConv}}
    scale::Float32
    nHeads::Int
    toQkv::T
    toOut::Chain{<:Tuple{T, GroupNorm}}
end

function LinearAttention{T}(dim::Int, nHeads::Int, dimHead::Int=32) where {T<:Union{Conv, CylindricalConv}}
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
    
    qkv = Flux.chunk(la.toQkv(x), 3; dims=4)
    q, k, v = map(t -> (@cast _[z⊗y⊗x, c, h, b] := t[z, y, x, c⊗h, b] h in 1:la.nHeads), qkv)

    q = softmax(q, dims=2) * la.scale
    softmax!(k, dims=1)
    
    @reduce c[e, d, h, b] := sum(n) k[n, d, h, b] * v[n, e, h, b]
    @reduce out[n, e, h, b] := sum(d) c[e, d, h, b] * q[n, d, h, b]

    la.toOut(@cast _[z, y, x, c⊗h, b] := out[z⊗y⊗x, c, h, b] h in 1:la.nHeads, z in 1:w, y in 1:h, x in 1:l)
end


struct PreNorm
    norm::GroupNorm
    layer
end

function PreNorm(dim::Int, layer::T) where {T}
    # layer must have AbstractArray functor 
    @assert any(method.sig >: Tuple{T, AbstractArray} for method in methods(layer)) "No functor for (layer::$T)(x::AbstractArray)"
    PreNorm(GroupNorm(dim, 1), layer)
end

Flux.@layer PreNorm

(pn::PreNorm)(x::AbstractArray) = (pn.layer ∘ pn.norm)(x)


struct Residual
    layer

    function Residual(layer::T) where {T}
        # layer must have AbstractArray functor 
        @assert any(method.sig >: Tuple{T, AbstractArray} for method in methods(layer)) "No functor for (layer::$T)(x::AbstractArray)"
        new(layer)
    end
end

Flux.@layer Residual

(res::Residual)(x::AbstractArray) = res.layer(x) + x