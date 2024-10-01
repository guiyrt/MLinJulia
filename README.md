# MLinJulia
GSoC @ CERN-HSF - "Machine Learning in Julia for Calorimeter Showers"

# Quick intro
This repo was made for the Julia implementation of CaloDiffusion (included as submodule). The normal installation proccess installs both. This is done via the `env.sh` script.

# Commands available
The script `env.sh` handles the most important actions you should need. These use as `./env.sh <action>` with the following actions:
- `install`: Installs conda enviroment for Python implementation of CaloDiffusion.
- `datasets`: Downloads datasets from CaloChallenge.
- `train`: Trains the model with dataset 2.
- `profile`: Runs benchmarks on Python and Julia implementations.
- `load`: Loads the environment. Use as `source env.sh load`.