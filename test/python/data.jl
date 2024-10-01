using PyCall, MLinJulia, Test, Underscores

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pyutils = pyimport("scripts.utils")

@testset "Dataset 2" begin
    c = TrainingConfig("configs/ds2_electron.yml")
    datasetpath = c.trainfiles[1]
    showers, energies = @_ MLinJulia.load_datafile(datasetpath) |> preprocess(c, __...)
    pyshowers, pyenergies = pyutils.DataLoader(file_name=datasetpath, shape=[-1, 1, 45, 16, 9], emax=1000.0, emin=1.0, nevts=-1,
                                            max_deposit=2, logE=true, showerMap="logit-norm", nholdout=0, dataset_num=2, orig_shape=false)

    @test showers ≈ reversedims(pyshowers)
    @test energies ≈ reversedims(pyenergies)
end

@testset "Dataset 3" begin
    c = TrainingConfig("configs/ds3_electron.yml")
    datasetpath = c.trainfiles[1]
    showers, energies = @_ MLinJulia.load_datafile(datasetpath) |> preprocess(c, __...)
    pyshowers, pyenergies = pyutils.DataLoader(file_name=datasetpath, shape=[-1, 1, 45, 50, 18], emax=1000.0, emin=1.0, nevts=-1,
                                            max_deposit=2, logE=true, showerMap="logit-norm", nholdout=0, dataset_num=3, orig_shape=false)

    @test showers ≈ reversedims(pyshowers)
    @test energies ≈ reversedims(pyenergies)
end