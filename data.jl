using HDF5, XML, IterTools, YAML


get_first_child(node) = node |> children |> first

# Document -> Bins -> Bin
bins = read("CaloChallenge/code/binning_dataset_1_photons.xml", LazyNode) |> get_first_child |> get_first_child

# Bin -> Layers
layers = bins |> children


pairwiseMidpoint(vec::Vector{Float64}) = [i+(j-i)/2 for (i, j) in partition(vec, 2, 1)]

struct CalorimeterLayer
    id::Int
    n_αBins::Int # Number of angular bins
    n_rBins::Int # Number of radial bins
    nBins::Int # Number of total bins
    rEdges::Vector{Float64} # Edge-point of radial bins
    rMidpoints::Vector{Float64} # Midpoint of radial bins
    αMidpoints::Vector{Float64} # Midpoint of angular bins

    function CalorimeterLayer(layer::Union{Node,LazyNode})
        attr = XML.attributes(layer)
        
        n_αBins = parse(Int, attr["n_bin_alpha"])
        rEdges = [parse(Float64, edge) for edge in split(attr["r_edges"], ',')]
        rMidpoints = pairwiseMidpoint(rEdges)
        αMidpoints = n_αBins > 1 ? LinRange(-π, π, n_αBins+1) |> collect |> pairwiseMidpoint : [0.]

        new(parse(Int, attr["id"]), n_αBins, length(rEdges)-1, n_αBins*n_rBins, rEdges, rMidpoints, αMidpoints)
    end
end


struct Calorimeter
    layers::Vector{CalorimeterLayer}
end



#logitTransform(showers::Matrix, α::Number, stats::Dict)

function loadData(filename::String, shape::Vector{Int64}, logE::Bool, eMax::Float64, eMin::Float64, maxDeposit::Int64, transforms::Vector, stats::Dict)
    @assert transforms[1] in ["sqrt", "logit", "log"] "Transformation '$(transforms[1])' not recognized. Expected 'sqrt', 'log' or 'logit'."
    @assert transforms[2] in ["norm", "scaled"] "Transformation '$(transforms[2])' not recognized. Expected 'norm' or 'scaled'."
    
    showers::Matrix{Float32} = h5read("datasets/$filename", "showers")
    energies::Matrix{Float32} = h5read("datasets/$filename", "incident_energies")

    showers ./= 1000.
    energies ./= 1000.

    showers ./= energies.*maxDeposit

    # Required to be defined as Float32 to have the same values as with numpy
    min::Float32, max::Float32, mean, std = stats[transforms[1]]["min"], stats[transforms[1]]["max"], stats[transforms[1]]["mean"], stats[transforms[1]]["std"]

    if transforms[1] == "sqrt"
        showers = sqrt.(showers)
    elseif transforms[1] == "logit"
        α = 1e-6
        showers = showers .* (1-2α) .+ α
        showers = log.(showers ./ (1 .- showers))
        replace!(showers, NaN => 0.)
    elseif transforms[1] == "log"
        showers = log.(showers)
        replace!(showers, NaN => min)
    end

    if transforms[2] == "norm"
        showers = (showers .- mean) ./ std
    elseif transforms[2] == "scaled"
        showers = transforms[1] == "sqrt" ? (showers.*2.0) .- 1.0 : 2.0 .* (showers.-min) ./ (max-min) .- 1.0
    end

    energies = logE ? log10.(energies./eMin) ./ log10(eMax/eMin) : (energies.-eMin) ./ (eMax-eMin)
       
    return showers, energies
end



a = YAML.load_file("ds2_electron.yml")
loadData(a["trainFiles"][1], a["shapePad"], a["logE"], a["eMax"], a["eMin"], a["maxDeposit"], a["showerTransforms"], a["stats"])