using Flux

const Device = Union{typeof(gpu), typeof(cpu)}
const ConvType = Union{Conv, CylindricalConv}