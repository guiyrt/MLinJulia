using PyCall, MLinJulia, Test

sys = pyimport("sys")
sys.path = push!(sys.path, "../CaloDiffusion")
pyModels = pyimport("scripts.models")

