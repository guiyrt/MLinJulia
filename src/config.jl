using YAML

struct TrainingConfig
    datasetFile::String
    trainFiles::Vector{String}
    evalFiles::Vector{String}
    showerShape::NTuple{4, Int}
    showerTransforms::Vector{String}
    trainValSplit::Float64
    batchSize::Int
    learningRate::Float64
    maxEpoch::Int
    earlyStop::Int
    layerSizeUnet::NTuple{4, Int}
    condSizeUnet::Int
    blockAttn::Bool
    midAttn::Bool
    compressZ:: Bool
    eMax::Float32
    eMin::Float32
    logE::Bool
    maxDeposit::Int
    cylindricalConv::Bool
    RZinput::Bool
    betaMax::Float64
    noiseSched::String
    nSteps::Int
    trainingObj::String
    lossType::String
    timeEmbed::String
    condEmbed::String
    sMean::Float32
    sStd::Float32
    sMin::Float32
    sMax::Float32

    function TrainingConfig(configpath::String)
        c = YAML.load_file(configpath)

        @assert c["showerTransforms"][1] in ["sqrt", "logit", "log"] "Transformation '$(transforms[1])' not recognized. Expected 'sqrt', 'log' or 'logit'."
        @assert c["showerTransforms"][2] in ["norm", "scaled"] "Transformation '$(transforms[2])' not recognized. Expected 'norm' or 'scaled'."

        new(
            c["datasetFile"],
            c["trainFiles"],
            c["evalFiles"],
            Tuple(c["showerShape"]),
            c["showerTransforms"],
            c["trainValSplit"],
            c["batchSize"],
            c["learningRate"],
            c["maxEpoch"],
            c["earlyStop"],
            Tuple(c["layerSizeUnet"]),
            c["condSizeUnet"],
            c["blockAttn"],
            c["midAttn"],
            c["compressZ"],
            convert(Float32, c["eMax"]),
            convert(Float32, c["eMin"]),
            c["logE"],
            c["maxDeposit"],
            c["cylindricalConv"],
            c["RZinput"],
            c["betaMax"],
            c["noiseSched"],
            c["nSteps"],
            c["trainingObj"],
            c["lossType"],
            c["timeEmbed"],
            c["condEmbed"],
            convert(Float32, c["stats"][c["showerTransforms"][1]]["mean"]),
            convert(Float32, c["stats"][c["showerTransforms"][1]]["std"]),
            convert(Float32, c["stats"][c["showerTransforms"][1]]["min"]),
            convert(Float32, c["stats"][c["showerTransforms"][1]]["max"])
        )
    end
end


#TODO const DS1π_CONFIG = TrainingConfig("configs/ds1_pion.yml")
#TODO const DS1γ_CONFIG = TrainingConfig("configs/ds1_photon.yml")
const DS2e⁻_CONFIG = TrainingConfig("configs/ds2_electron.yml")
#TODO const DS3e⁻_CONFIG = TrainingConfig("configs/ds3_electron.yml")