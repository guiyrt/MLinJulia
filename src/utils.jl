fn_if(fn, b::Bool) = b ? fn : identity

reversedims(x::AbstractArray) = permutedims(x, ndims(x):-1:1)