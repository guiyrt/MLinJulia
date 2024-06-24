using Flux, PyCall


function setparams!(cd::CaloDiffusion, torchcd::PyObject)
    @assert isclassname(torchcd, "CondUnet")

    setparams!(cd.timenet, torchcd.time_mlp)
    setparams!(cd.condnet, torchcd.cond_mlp)

    for (torchblock, torchattn, block) in zip(torchcd.downs, torchcd.downs_attn, cd.downs)
        setparams!(block.resnetblock₁, get(torchblock, 0))
        setparams!(block.resnetblock₂, get(torchblock, 1))
        setparams!(block.blockattention, torchattn)
        setparams!(block.downsampler, get(torchblock, 2))
    end

    setparams!(cd.mid.resnetblock₁, torchcd.mid_block1)
    setparams!(cd.mid.midattention, torchcd.mid_attn)
    setparams!(cd.mid.resnetblock₂, torchcd.mid_block2)

    for (torchblock, torchattn, block) in zip(torchcd.ups, torchcd.ups_attn, cd.ups)
        setparams!(block.resnetblock₁, get(torchblock, 0))
        setparams!(block.resnetblock₂, get(torchblock, 1))
        setparams!(block.blockattention, torchattn)
        setparams!(block.upsampler, get(torchblock, 2))
    end

    setparams!(cd.mid.resnetblock₁, torchcd.mid_block1)
    setparams!(cd.mid.midattention, torchcd.mid_attn)
    setparams!(cd.mid.resnetblock₂, torchcd.mid_block2)

    setparams!(cd.outconv, torchcd.final_conv)
end

function CaloDiffusion(torchcd::PyObject)
    @assert isclassname(torchcd, "CondUnet")

    convtype = isclassname(torchcd.init_conv, "CylindricalConv") ? CylindricalConv : Conv
    embeddingstype = isclassname(torchcd.time_mlp, "MlpEmbeddings") ? MlpEmbeddings : SinusoidalPositionEmbeddings

    CaloDiffusion{convtype}(
        embeddingstype(torchcd.time_mlp),
        embeddingstype(torchcd.cond_mlp),
        convtype(torchcd.init_conv),
        [
            MLinJulia.DownBlock(
                ResNetBlock(get(downblock, 0)),
                ResNetBlock(get(downblock, 1)),
                Residual(down_attention),
                frompyclass(get(downblock, 2))
            ) for (downblock, down_attention) in zip(torchcd.downs, torchcd.downs_attn)
        ],
        MLinJulia.MidBlock(ResNetBlock(torchcd.mid_block1), Residual(torchcd.mid_attn), ResNetBlock(torchcd.mid_block2)),
        [
            MLinJulia.UpBlock(
                ResNetBlock(get(upblock, 0)),
                ResNetBlock(get(upblock, 1)),
                Residual(up_attention),
                frompyclass(get(upblock, 2))
            ) for (upblock, up_attention) in zip(torchcd.ups, torchcd.ups_attn)
        ],
        Chain(torchcd.final_conv)
    )
end