using PyCall


function SinusoidalPositionEmbeddings(torchspe::PyObject)
    @assert isclassname(torchspe, "SinusoidalPositionEmbeddings")
    
    SinusoidalPositionEmbeddings(torchspe.dim_in, Dense(get(torchspe.hidden, 0), gelu), Dense(torchspe.out))
end

function setparams!(spe::SinusoidalPositionEmbeddings, torchspe::PyObject)
    setparams!(spe.hidden, get(torchspe.hidden, 0))
    setparams!(spe.out, torchspe.out)
end


function MlpEmbeddings(torchmlpe::PyObject)
    @assert isclassname(torchmlpe, "MlpEmbeddings")

    MlpEmbeddings(Dense(get(torchmlpe._in, 0), gelu), Dense(get(torchmlpe.hidden, 0), gelu), Dense(torchmlpe.out))
end

function setparams!(mlpe::MlpEmbeddings, torchmlpe::PyObject)
    @assert isclassname(torchmlpe, "MlpEmbeddings")

    setparams!(mlpe.in, get(torchmlpe._in, 0))
    setparams!(mlpe.hidden, get(torchmlpe.hidden, 0))
    setparams!(mlpe.out, torchmlpe.out)
end