using YAML

struct TrainingConfig
    calo::Calorimeter
    trainfiles::Vector{String}
    testfiles::Vector{String}
    shower_transforms::Vector{String}
    train_val_split::Float64
    batchsize::Int
    learning_rate::Float64
    epochs::Int
    earlystop::Int
    blocksize_unet::Vector{Int}
    e_max::Float32
    e_min::Float32
    e_log::Bool
    maxdeposit::Int
    cylindricalconv::Bool
    ϕ_input::Bool
    beta_max::Float64
    nsteps::Int
    noise_pred_loss::Bool
    s_mean::Float32
    s_std::Float32
    s_min::Float32
    s_max::Float32

    function TrainingConfig(configpath::String)
        c = YAML.load_file(configpath)

        @assert c["shower_transforms"][1] in ["sqrt", "logit", "log"] "Transformation '$(transforms[1])' not recognized. Expected 'sqrt', 'log' or 'logit'."
        @assert c["shower_transforms"][2] in ["norm", "scaled"] "Transformation '$(transforms[2])' not recognized. Expected 'norm' or 'scaled'."

        new(
            Calorimeter(c["binningfile"]),
            c["trainfiles"],
            c["testfiles"],
            c["shower_transforms"],
            c["train_val_split"],
            c["batchsize"],
            c["learning_rate"],
            c["epochs"],
            c["earlystop"],
            c["blocksize_unet"],
            convert(Float32, c["e_max"]),
            convert(Float32, c["e_min"]),
            c["e_log"],
            c["maxdeposit"],
            c["cylindricalconv"],
            c["phi_input"],
            c["beta_max"],
            c["nsteps"],
            c["noise_pred_loss"],
            convert(Float32, c["stats"][c["shower_transforms"][1]]["mean"]),
            convert(Float32, c["stats"][c["shower_transforms"][1]]["std"]),
            convert(Float32, c["stats"][c["shower_transforms"][1]]["min"]),
            convert(Float32, c["stats"][c["shower_transforms"][1]]["max"])
        )
    end
end


#TODO const DS1π_CONFIG = TrainingConfig("configs/ds1_pion.yml")
#TODO const DS1γ_CONFIG = TrainingConfig("configs/ds1_photon.yml")
const DS2e⁻_CONFIG = TrainingConfig("configs/ds2_electron.yml")
#TODO const DS3e⁻_CONFIG = TrainingConfig("configs/ds3_electron.yml")