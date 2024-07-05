using Flux, PyCall


function setparams!(unet::CondUnet, torchunet::PyObject)
    @assert isclassname(torchunet, "CondUnet")

    setparams!(unet.timenet, torchunet.time_mlp)
    setparams!(unet.condnet, torchunet.cond_mlp)

    for (torchblock, torchattn, block) in zip(torchunet.downs, torchunet.downs_attn, unet.downs)
        setparams!(block.resnetblock₁, get(torchblock, 0))
        setparams!(block.resnetblock₂, get(torchblock, 1))
        setparams!(block.blockattention, torchattn)
        setparams!(block.downsampler, get(torchblock, 2))
    end

    setparams!(unet.mid.resnetblock₁, torchunet.mid_block1)
    setparams!(unet.mid.midattention, torchunet.mid_attn)
    setparams!(unet.mid.resnetblock₂, torchunet.mid_block2)

    for (torchblock, torchattn, block) in zip(torchunet.ups, torchunet.ups_attn, unet.ups)
        setparams!(block.resnetblock₁, get(torchblock, 0))
        setparams!(block.resnetblock₂, get(torchblock, 1))
        setparams!(block.blockattention, torchattn)
        setparams!(block.upsampler, get(torchblock, 2))
    end

    setparams!(unet.mid.resnetblock₁, torchunet.mid_block1)
    setparams!(unet.mid.midattention, torchunet.mid_attn)
    setparams!(unet.mid.resnetblock₂, torchunet.mid_block2)

    setparams!(unet.outconv, torchunet.final_conv)
end

function CondUnet(torchunet::PyObject)
    @assert isclassname(torchunet, "CondUnet")

    convtype = isclassname(torchunet.init_conv, "CylindricalConv") ? CylindricalConv : Conv
    embeddingstype = isclassname(torchunet.time_mlp, "MlpEmbeddings") ? MlpEmbeddings : SinusoidalPositionEmbeddings

    downs = [
        MLinJulia.DownBlock(
            ResNetBlock(get(downblock, 0)),
            ResNetBlock(get(downblock, 1)),
            Residual(down_attention),
            frompyclass(get(downblock, 2))
        ) for (downblock, down_attention) in zip(torchunet.downs, torchunet.downs_attn)
    ]

    ups = [
        MLinJulia.UpBlock(
            ResNetBlock(get(upblock, 0)),
            ResNetBlock(get(upblock, 1)),
            Residual(up_attention),
            frompyclass(get(upblock, 2))
        ) for (upblock, up_attention) in zip(torchunet.ups, torchunet.ups_attn)
    ]

    CondUnet{convtype}(
        embeddingstype(torchunet.time_mlp),
        embeddingstype(torchunet.cond_mlp),
        convtype(torchunet.init_conv),
        [MLinJulia.UnetLayer{convtype}(down, up) for (down, up) in zip(downs, reverse(ups))],
        MLinJulia.MidBlock(ResNetBlock(torchunet.mid_block1), Residual(torchunet.mid_attn), ResNetBlock(torchunet.mid_block2)),
        Chain(torchunet.final_conv)
    )
end