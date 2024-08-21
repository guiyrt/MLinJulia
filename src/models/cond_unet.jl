using Flux, TensorCast, Underscores, IterTools


struct DownBlock{T<:Union{Conv, CylindricalConv}}
    resnetblock₁::ResNetBlock{T}
    resnetblock₂::ResNetBlock{T}
    blockattention::Residual
    downsampler::Union{T, typeof(identity)}
end

struct MidBlock{T<:Union{Conv, CylindricalConv}}
    resnetblock₁::ResNetBlock{T}
    midattention::Residual
    resnetblock₂::ResNetBlock{T}
end

struct UpBlock{T<:Union{Conv, CylindricalConv}}
    resnetblock₁::ResNetBlock{T}
    resnetblock₂::ResNetBlock{T}
    blockattention::Residual
    upsampler::Union{Union{ConvTranspose, CylindricalConvTranspose}, typeof(identity)}
end

mutable struct UnetLayer{T<:Union{Conv, CylindricalConv}}
    downblock::DownBlock{T}
    upblock::UpBlock{T}
    skipconn::Union{Nothing, AbstractArray}
end

function UnetLayer{T}(downblock::DownBlock{T}, upblock::UpBlock{T}) where {T<:Union{Conv, CylindricalConv}}
    UnetLayer{T}(downblock, upblock, nothing)
end

function DownBlock{T}(dim_in::Int, dim_out::Int, downsample::Bool; cond_dim::Int = 128, resnetblockgroups:: Int = 8) where {T<:Union{Conv, CylindricalConv}}
    DownBlock{T}(
        ResNetBlock{T}(dim_in, dim_out, resnetblockgroups, cond_dim),
        ResNetBlock{T}(dim_out, dim_out, resnetblockgroups, cond_dim),
        (@_ LinearAttention{T}(dim_out) |> PreNorm(dim_out, __) |> Residual),
        downsample ? T((4,4,3), dim_out => dim_out; stride=(2, 2, 2), pad=1) : identity
    )
end

function MidBlock{T}(dim::Int, cond_dim::Int = 128, resnetblockgroups:: Int = 8) where {T<:Union{Conv, CylindricalConv}}
    MidBlock{T}(
        ResNetBlock{T}(dim, dim, resnetblockgroups, cond_dim),
        (@_ LinearAttention{T}(dim) |> PreNorm(dim, __) |> Residual),
        ResNetBlock{T}(dim, dim, resnetblockgroups, cond_dim)
    )
end

function UpBlock{T}(dim_in::Int, dim_out::Int, outpad::Union{NTuple{3, Int}, Nothing};
                    cond_dim::Int = 128, resnetblockgroups:: Int = 8) where {T<:Union{Conv, CylindricalConv}}
    transposetype = T === CylindricalConv ? CylindricalConvTranspose : ConvTranspose
    UpBlock{T}(
        ResNetBlock{T}(dim_in*2, dim_out, resnetblockgroups, cond_dim),
        ResNetBlock{T}(dim_out, dim_out, resnetblockgroups, cond_dim),
        (@_ LinearAttention{T}(dim_out) |> PreNorm(dim_out, __) |> Residual),
        !isnothing(outpad) ? transposetype((4, 4, outpad[3] > 0 ? 4 : 3), dim_out => dim_out; stride=(2, 2, 2), pad=1, outpad=outpad) : identity
    )
end

Flux.@layer DownBlock
Flux.@layer MidBlock
Flux.@layer UpBlock
Flux.@layer UnetLayer

function (db::DownBlock)(x::AbstractArray, cond::AbstractArray)
    attention_out = @_ x |> db.resnetblock₁(__, cond) |> db.resnetblock₂(__, cond) |> db.blockattention
    return attention_out, db.downsampler(attention_out)
end

(mb::MidBlock)(x::AbstractArray, cond::AbstractArray) = @_ mb.resnetblock₁(x, cond) |> mb.midattention(__) |> mb.resnetblock₂(__, cond)

function (ub::UpBlock)(x::AbstractArray, cond::AbstractArray, skipconn::AbstractArray)
    @_ cat(x, skipconn; dims=4) |> ub.resnetblock₁(__, cond) |> ub.resnetblock₂(__, cond) |> ub.blockattention |> ub.upsampler
end

function (unetl::UnetLayer)(x::AbstractArray, cond::AbstractArray)
    if isnothing(unetl.skipconn)
        skipconn, out = unetl.downblock(x, cond)
        unetl.skipconn = skipconn
    else
        out = unetl.upblock(x, cond, unetl.skipconn)
        unetl.skipconn = nothing
    end

    return out
end


struct CondUnet{T<:Union{Conv, CylindricalConv}}
    timenet::Union{MlpEmbeddings, SinusoidalPositionEmbeddings}
    condnet::Union{MlpEmbeddings, SinusoidalPositionEmbeddings}
    inconv::T
    layers::Vector{<:UnetLayer}
    mid::MidBlock
    outconv::Chain{Tuple{ResNetBlock{T}, T}}
end

function CondUnet{T}(showershape::NTuple{3, Int}, inchannels::Int, blocksizes::Vector{Int};
                     cond_dim::Int = 128, resnetblockgroups:: Int = 8, sinusoidal_embeddings::Bool = false
                    ) where {T<:Union{Conv, CylindricalConv}}
    layers::Vector{UnetLayer} = []

    blockdims = partition(blocksizes, 2, 1)
    nblocks = length(blockdims)
    
    blockoutpad = nothing
    blockshowershape = showershape
    
    for (i, (dim_in, dim_out)) in enumerate(blockdims)
        push!(layers, UnetLayer{T}(
            DownBlock{T}(dim_in, dim_out, i != nblocks; cond_dim, resnetblockgroups),
            UpBlock{T}(dim_out, dim_in, blockoutpad; cond_dim, resnetblockgroups)
            )
        )

        blockoutpad = (blockshowershape .+ (0, 0, 1)) .% 2
        blockshowershape = (blockshowershape[1]÷2, blockshowershape[2]÷2, ceil(Int, blockshowershape[3]/2))
    end

    CondUnet{T}(
        sinusoidal_embeddings ? SinusoidalPositionEmbeddings(cond_dim) : MlpEmbeddings(cond_dim),
        sinusoidal_embeddings ? SinusoidalPositionEmbeddings(cond_dim) : MlpEmbeddings(cond_dim),
        T((3,3,3), inchannels => blocksizes[1]; pad=1),
        layers,
        MidBlock{T}(blocksizes[end], cond_dim, resnetblockgroups),
        Chain(ResNetBlock{T}(blocksizes[2], blocksizes[3], resnetblockgroups), T((1,1,1), blocksizes[3] => 1))
    )
end

Flux.@layer CondUnet

function (m::CondUnet)(x::AbstractArray, cond::AbstractArray, time::AbstractArray)
    conds = cat(m.timenet(time), m.condnet(cond); dims=1)
    
    reduceblocks = (out, block) -> block(out, conds)
    @_ x |> m.inconv |> reduce(reduceblocks, m.layers; init=__) |> m.mid(__, conds) |> reduce(reduceblocks, reverse(m.layers); init=__) |> m.outconv
end