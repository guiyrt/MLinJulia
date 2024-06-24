using PyCall, Flux, Test, CUDA
nn = pyimport("torch.nn")

@testset "ConvTranspose" begin
    # Random input
    data, torchdata = rand32tensors(4, 8, 23, 16, 128)

    torchcc = nn.ConvTranspose3d(16, 16, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1,3,1), output_padding=1)
    cc = ConvTranspose(torchcc)

    # Identical output
    @test cc(data) ≈ torchcc(torchdata) |> fromtorchtensor
end