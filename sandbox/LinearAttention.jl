using PyCall, TensorCast, CUDA, Flux, MLinJulia
einops = pyimport("einops")
torch = pyimport("torch")
include("../test/torch/utils.jl")

togpu=true

inputSize = (128, 32, 45, 16, 9)
kvqSize = (6480, 32, 1, 128)
data, torchdata = rand32tensors(inputSize...; togpu);

_, _, w, h, l = inputSize
nHeads = 1


inputreshape(x) = @cast _[z⊗y⊗x, c, h, b] := x[z, y, x, c⊗h, b] h in 1:nHeads;
einsum_one(k,v) = @reduce _[e, d, h, b] := sum(n) k[n, d, h, b] * v[n, e, h, b];
einsum_two(c,q) = @reduce _[n, e, h, b] := sum(d) c[e, d, h, b] * q[n, d, h, b];
outputreshape(x) = @cast _[z, y, x, c⊗h, b] := x[z⊗y⊗x, c, h, b] h in 1:nHeads, x in 1:l, y in 1:h, z in 1:w;

println("Input reshape")
pyqkv = einops.rearrange(torchdata, "b (h c) x y z -> b h c (x y z)", h=nHeads);
CUDA.@time qkv = inputreshape(data);
CUDA.@time qkv = inputreshape(data);
@assert qkv |> cpu ≈ pyqkv |> fromtorchtensor

println("Einsum one")
k, pyk = rand32tensors(kvqSize...; togpu);
v, pyv = rand32tensors(kvqSize...; togpu);
pyc = torch.einsum("b h d n, b h e n -> b h d e", pyk, pyv);
CUDA.@time c = einsum_one(k,v)
CUDA.@time c = einsum_one(k,v)
@assert c |> cpu ≈ pyc |> fromtorchtensor

println("Einsum two")
q, pyq = rand32tensors(kvqSize...; togpu);
pyout = torch.einsum("b h d e, b h d n -> b h e n", pyc, pyq);
CUDA.@time out = einsum_two(c, q)
CUDA.@time out = einsum_two(c, q)
@assert out |> cpu ≈ pyout |> fromtorchtensor

println("Output reshape")
pyoutput = einops.rearrange(pyout, "b h c (x y z) -> b (h c) x y z", h=nHeads, x=l, y=h, z=w);
CUDA.@time output = outputreshape(out)
CUDA.@time output = outputreshape(out)
@assert output |> cpu ≈ pyoutput |> fromtorchtensor