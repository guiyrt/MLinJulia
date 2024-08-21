#!/bin/bash
set -e


# Run command
case "$1" in
    # Create environments and install dependencies
    install)
        # Install julia if not installed
        if ! command -v julia; then
            wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz
            tar zxvf julia-1.10.4-linux-x86_64.tar.gz -C "$HOME"
            rm julia-1.10.4-linux-x86_64.tar.gz
            echo 'export PATH="$PATH:$HOME/julia-1.10.4/bin"' >> $HOME/.bashrc
        fi

        # Checkout to submodules' tracked version (also clones submodules if needed)
        git submodule update --init

        # Create conda environment for original implementation
        conda env create -f CaloDiffusion/conda.yml
        
        exit 0
        ;;

    # Load python venv
    load)
        set +e
        conda activate CaloDiffusion
        ;;

    # Benchmark training loop
    benchmark)
        [ -d "benchmarks" ] || mkdir "benchmarks"
        commit_hash=$(git rev-parse --short HEAD)
        timestamp=$(date +%d%m%Y)
        julia --project src/calodiffusion/benchmark.jl > "benchmarks/${timestamp}_${commit_hash}.txt"
        
        exit 0
        ;;

    # Remove environment
    remove)
        conda env remove -n CaloDiffusion
        
        exit 0
        ;;

    datasets)
        # Download and extract datasets
        if [ ! -d "datasets" ]; then
            mkdir datasets
            cd datasets

            # All 3 datasets
            for doi in 8099322 6366271 6366324; do
                wget "https://zenodo.org/api/records/$doi/files-archive" -O dataset.zip
                unzip dataset.zip
                rm dataset.zip
            done

            chmod 555 *
        fi

        exit 0
        ;;


    # Print help message
    help|*)
        [ "$1" != "help" ] && echo "Unknown command '$1'"

        echo "Usage: ./env.sh COMMAND"
        echo "Commands:"
        echo "  install    | Installs submodules in python venv"
        echo "  benchmark  | Save to file benchmark results for training loop"
        echo "  remove     | Removes python venv"
        echo "  load       | Activates python venv (use with source)"
        echo "  datasets   | Downloads datasets"
        echo "  help       | Print this help message"
        ;;
esac