using PyCall, MLinJulia, Flux, Test
include("utils.jl")

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pyModels = pyimport("scripts.models")
np = pyimport("numpy")
torch = pyimport("torch")
nn = pyimport("torch.nn")


@testset "CylindricalConvTrans" begin
    # Random input
    data = rand32(1, 16, 12, 4, 2)
    torchData = torch.Tensor(data)

    # CylindricalConvTranspose layers
    torchCct = pyModels.CylindricalConvTranspose(16, 16, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=1)
    cct = CylindricalConvTranspose((4,4,3), 16=>16, stride=(2, 2, 2), pad=1)

    # Set all weights to 1
    w, b = torchCct.parameters()
    w.data = nn.parameter.Parameter(torch.ones_like(w))
    w = w.detach().numpy()
    b = b.detach().numpy()

    # Same number of trainable parameters
    @test (nparams(cct.convTranspose.weight) + nparams(cct.convTranspose.bias)) == (nparams(w) + nparams(b))
    
    # Flux layer with the same weights and bias as PyTorch layer
    cct.convTranspose.weight .= w |> reversedims
    cct.convTranspose.bias .= b

    # Forward pass
    pyOut = torchCct(torchData).detach().numpy() |> reversedims
    out = data |> reversedims |> cct

    # Nearly identical output
    @test all(out .≈ pyOut)
end