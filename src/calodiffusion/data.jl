function createϕ_image(shape::NTuple{4, Int})
    step = 1f0 / (shape[2]-1)
    binvalues = [i*step for i in 0:shape[2]-1]
    binsize = (shape[1], 1, shape[3:end]...)

    return cat([fill(v, binsize) for v in binvalues]...; dims=2)
end

function createR_image(shape::NTuple{4, Int}, r_midpoints::Vector{Float32})
    imageshape = (1, shape[2:end]...)
    image = cat([fill(v, imageshape) for v in r_midpoints]...; dims=1)
    
    return image ./ r_midpoints[end]
end

function createZ_image(shape::NTuple{4, Int})
    imageshape = (shape[1:2]..., 1, shape[4:end]...)
    image = cat([fill(v, imageshape) for v in 0f0:shape[3]-1]...; dims=3)

    return image ./ shape[3]
end

createRZ_images(shape::NTuple{4, Int}, r_midpoints) = cat(createR_image(shape, r_midpoints), createZ_image(shape); dims=4)
createRZϕ_images(shape::NTuple{4, Int}, r_midpoints) = cat(createRZ_images(shape, r_midpoints), createϕ_image(shape); dims=4)


function extract(values::AbstractArray{Float32}, indices::AbstractArray{Int})
    reshape(values[indices], (1, 1, 1, 1, :))
end


function applynoise(sched::CosineSchedule, x::AbstractArray{Float32, 5}, noise::AbstractArray{Float32, 5}, t::AbstractArray{Int, 1})
        return (extract(sched.sqrt_αcumprod, t) .* x) + (extract(sched.sqrt_one_minus_αcumprod, t) .* noise)
end