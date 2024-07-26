using XML, IterTools, OrderedCollections


getfirstchild(node::Union{Node,LazyNode}) = node |> children |> first

midpoints(vec) = [(i+j)/2 for (i, j) in partition(vec, 2, 1)]


struct CalorimeterLayer
    id::Int
    n_αbins::Int # Number of angular bins
    r_edges::Vector{Float64} # Edge-point of radial bins

    function CalorimeterLayer(layer::Union{Node, LazyNode})
        attr = XML.attributes(layer)
        
        new(
            parse(Int, attr["id"]),
            parse(Int, attr["n_bin_alpha"]),
            [parse(Float32, edge) for edge in split(attr["r_edges"], ',')]
        )
    end
end


struct Calorimeter
    particle::String
    layers::Vector{CalorimeterLayer}
    r_midpoints::Vector{Float32}
    α_midpoints::Vector{Float64}
    shape::NTuple{4, Int}
    nvoxels::Int

    function Calorimeter(binningfile::String)
        bins = read(binningfile, LazyNode) |> getfirstchild |> getfirstchild
        layers = [CalorimeterLayer(layer) for layer in children(bins)]

        r_edges = reduce(layers[2:end]; init=OrderedSet(layers[1].r_edges)) do edges, layer
            union(edges, OrderedSet(layer.r_edges))
        end |> sort
        r_midpoints = midpoints(r_edges)

        n_αbins = maximum(layer.n_αbins for layer in layers)
        α_midpoints = LinRange(-π, π, n_αbins+1) |> midpoints

        shape = (length(r_midpoints), n_αbins , length(layers), 1)

        return new(
            XML.attributes(bins)["name"],
            layers,
            r_midpoints,
            α_midpoints,
            shape,
            prod(shape)
        )
    end
end