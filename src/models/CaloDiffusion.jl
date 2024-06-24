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

function DownBlock{T}(dim_in::Int, dim_out::Int, downsample::Bool; cond_dim::Int = 128, resnetblockgroups:: Int = 8) where {T<:Union{Conv, CylindricalConv}}
    DownBlock{T}(
        ResNetBlock{T}(dim_in, dim_out, resnetblockgroups, cond_dim),
        ResNetBlock{T}(dim_out, dim_out, resnetblockgroups, cond_dim),
        (@_ LinearAttention{T}(dim_out) |> PreNorm(dim_out, __) |> Residual),
        downsample ? T((4,4,3), dim_out => dim_out; stride=(2, 2, 2), pad=1) : identity
    )
end

function MidBlock{T}(dim::Int; cond_dim::Int = 128, resnetblockgroups:: Int = 8) where {T<:Union{Conv, CylindricalConv}}
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

function (db::DownBlock)(x::AbstractArray, cond::AbstractArray, skipconn::Vector{AbstractArray})
    attention_out = @_ x |> db.resnetblock₁(__, cond) |> db.resnetblock₂(__, cond) |> db.blockattention
    push!(skipconn, attention_out)
    db.downsampler(attention_out)
end

(mb::MidBlock)(x::AbstractArray, cond::AbstractArray) = @_ mb.resnetblock₁(x, cond) |> mb.midattention(__) |> mb.resnetblock₂(__, cond)

function (ub::UpBlock)(x::AbstractArray, cond::AbstractArray, skipconn::Vector{AbstractArray})
    @_ cat(x, pop!(skipconn); dims=4) |> ub.resnetblock₁(__, cond) |> ub.resnetblock₂(__, cond) |> ub.blockattention |> ub.upsampler
end


struct CaloDiffusion{T<:Union{Conv, CylindricalConv}}
    timenet::Union{MlpEmbeddings, SinusoidalPositionEmbeddings}
    condnet::Union{MlpEmbeddings, SinusoidalPositionEmbeddings}
    inconv::T
    downs::Vector{DownBlock}
    mid::MidBlock
    ups::Vector{UpBlock}
    outconv::Chain{Tuple{ResNetBlock{T}, T}}
end

function CaloDiffusion{T}(showershape::NTuple{3, Int}, inchannels::Int, blocksizes::Vector{Int};
                          cond_dim::Int = 8, resnetblockgroups:: Int = 8, sinusoidal_embeddings::Bool = false) where {T<:Union{Conv, CylindricalConv}}
    upblocks::Vector{UpBlock} = []
    downblocks::Vector{DownBlock} = []

    blockdims = partition(blocksizes, 2, 1)
    nblocks = length(blockdims)
    
    blockoutpad = nothing
    blockshowershape = showershape
    
    for (i, (dim_in, dim_out)) in enumerate(blockdims)
        push!(downblocks, DownBlock{T}(dim_in, dim_out, i != nblocks; cond_dim, resnetblockgroups))
        push!(upblocks, UpBlock{T}(dim_out, dim_in, blockoutpad; cond_dim, resnetblockgroups))

        blockoutpad = (blockshowershape .+ (0, 0, 1)) .% 2
        blockshowershape = (blockshowershape[1]÷2, blockshowershape[2]÷2, ceil(Int, blockshowershape[3]/2))
    end

    CaloDiffusion{T}(
        sinusoidal_embeddings ? SinusoidalPositionEmbeddings(cond_dim) : MlpEmbeddings(cond_dim),
        sinusoidal_embeddings ? SinusoidalPositionEmbeddings(cond_dim) : MlpEmbeddings(cond_dim),
        T((3,3,3), inchannels => blocksizes[3]; pad=1),
        downblocks,
        MidBlock{T}(blocksizes[end]; cond_dim, resnetblockgroups),
        reverse(upblocks),
        Chain(ResNetBlock{T}(blocksizes[2], blocksizes[3], cond_dim, nothing), T((1,1,1), blocksizes[3] => 1))
    )
end

Flux.@layer CaloDiffusion

function (cd::CaloDiffusion)(x::AbstractArray, cond::AbstractArray, time::AbstractArray)
    conds = cat(cd.timenet(time), cd.condnet(cond); dims=1)
    
    skipconn::Vector{AbstractArray} = []
    reduceblocks = (out, block) -> block(out, conds, skipconn)
    @_ x |> cd.inconv |> reduce(reduceblocks, cd.downs; init=__) |> cd.mid(__, conds) |> reduce(reduceblocks, cd.ups; init=__) |> cd.outconv
end