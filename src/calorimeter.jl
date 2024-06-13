using XML, IterTools


getfirstchild(node::Union{Node,LazyNode}) = node |> children |> first

# Document -> Bins -> Bin
bins = read("CaloChallenge/code/binning_dataset_1_photons.xml", LazyNode) |> getfirstchild |> getfirstchild

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