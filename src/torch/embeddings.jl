using PyCall


function SinusoidalPositionEmbeddings(torchspe::PyObject)
    @assert isclassname(torchspe, "SinusoidalPositionEmbeddings")
    
    SinusoidalPositionEmbeddings(torchspe.dim)
end