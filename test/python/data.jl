using PyCall, MLinJulia, Test

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pyutils = pyimport("scripts.utils")

@testset "Dataset file loading and pre-processing" begin
    showers, inergies = MLinJulia.loadDataFile(DS2e⁻_CONFIG, "../datasets/dataset_2_1.hdf5")
    pyshowers, pyinergies = pyutils.DataLoader(file_name="../datasets/dataset_2_1.hdf5", shape=[-1, 1, 45, 16, 9], emax=1000.0, emin=1.0, nevts=-1,
                                            max_deposit=2, logE=true, showerMap="logit-norm", nholdout=0, dataset_num=2, orig_shape=false)

    @test showers ≈ reversedims(pyshowers)
    @test inergies ≈ reversedims(pyinergies)
end