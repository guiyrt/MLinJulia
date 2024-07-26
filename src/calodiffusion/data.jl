function createϕ_image(shape::NTuple{4, Int}, batchsize::Int)
    step = 1f0 / (shape[2]-1)
    binvalues = [i*step for i in 0:shape[2]-1]

    binsize = (shape[1], 1, shape[3:end]...)
    image = cat([fill(v, binsize) for v in binvalues]...; dims=2)

    return repeat(image, 1, 1, 1, 1, batchsize)
end

function createR_image(shape::NTuple{4, Int}, batchsize::Int, r_midpoints::Vector{Float32})
    imageshape = (1, shape[2:end]...)
    image = cat([fill(v, imageshape) for v in r_midpoints]...; dims=1)
    
    return repeat(image ./ r_midpoints[end], 1, 1, 1, 1, batchsize)
end

function createZ_image(shape::NTuple{4, Int}, batchsize::Int)
    imageshape = (shape[1:2]..., 1, shape[4:end]...)
    image = cat([fill(v, imageshape) for v in 0f0:shape[3]-1]...; dims=3)

    return repeat(image ./ shape[3], 1, 1, 1, 1, batchsize)
end

createRZ_images(shape::NTuple{4, Int}, batchsize::Int, r_midpoints) = cat(createR_image(shape, batchsize, r_midpoints), createZ_image(shape, batchsize); dims=4)
createRZϕ_images(shape::NTuple{4, Int}, batchsize::Int, r_midpoints) = cat(createRZ_images(shape, batchsize, r_midpoints), createϕ_image(shape, batchsize); dims=4)


extract(values::Vector, indices::Vector{Int}) = @_ NNlib.gather(values, indices) |> reshape(__, (1, 1, 1, 1, :))


function applynoise(sched::CosineSchedule, x::T, noise::T, t::Vector{Int}) where {T<:AbstractArray}
    t[1] == 1 ? x : extract(sched.sqrt_αcumprod, t) .* x + extract(sched.sqrt_one_minus_αcumprod, t) .* noise
end