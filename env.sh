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
        ;;

    # Load python venv
    load)
        set +e
        conda activate CaloDiffusion
        ;;

    # Benchmark training loop
    profile)
        set +e
        # Check system paranoid level
        sys_paranoia=$(sysctl -n kernel.perf_event_paranoid)
        if [ "$sys_paranoia" -gt 1 ]; then
            echo "NVIDIA Nsight Systems requires system's paranoid level to be 1 or lower for CPU IP/backtrace sampling (currently $sys_paranoia)."
            echo "Set kernel paranoid level to 1 with 'sudo sysctl kernel.perf_event_paranoid=1'."
            exit 1
        fi

        # Create folder to save profiling data based on date and commit hash
        reports_folder="$(pwd)/profiler/$(date +%Y-%m-%dT%H:%M:%S)_$(git rev-parse --short HEAD)"
        mkdir -p $reports_folder

        julia_calodif_dir=$(pwd)
        python_calodif_dir=$(pwd)/CaloDiffusion

        for batch_size in 4 16 32; do
            # Julia
            cd "$julia_calodif_dir"
            for device in cpu gpu; do
                report_name="$reports_folder/${device}_${batch_size}_jl"
                
                # NVIDIA Nsights Systems
                nsys launch --trace cuda,nvtx,cublas,cudnn,osrt --cuda-memory-usage true julia --project src/calodiffusion/profile_cuda.jl configs/ds2_electron.yml --batchsize "$batch_size" --device "$device"
                mv report1.nsys-rep "${report_name}_nsys.nsys-rep"
                nsys analyze "${report_name}_nsys.nsys-rep" --timeunit msec > "${report_name}_nsys.txt"

                # Julia BenchmarkTools
                julia --project src/calodiffusion/benchmark.jl configs/ds2_electron.yml --device "$device" --batchsize "$batch_size" --output "${report_name}_benchmark"
            done

            # Python
            cd "$python_calodif_dir"
            for device in cpu cuda; do
                report_name="$reports_folder/${device}_${batch_size}_py"
                
                # NVIDIA Nsights Systems
                nsys profile --output "${report_name}_nsys" --capture-range cudaProfilerApi --trace cuda,nvtx,cublas,cudnn,osrt --cuda-memory-usage true python -m scripts.profile_cuda configs/config_dataset2.json --batch-size "$batch_size" --device "$device"
                nsys analyze "${report_name}_nsys.nsys-rep" --timeunit msec > "${report_name}_nsys.txt"
                
                # PyTorch Benchmarker
                python -m scripts.benchmark configs/config_dataset2.json --batch-size "$batch_size" --device "$device" --output "${report_name}_benchmark"
            done
        done
        ;;

    # Remove environment
    remove)
        conda env remove -n CaloDiffusion
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
        ;;


    # Print help message
    help|*)
        [ "$1" != "help" ] && echo "Unknown command '$1'"

        echo "Usage: ./env.sh COMMAND"
        echo "Commands:"
        echo "  install    | Installs CaloDiffusion conda enviroment"
        echo "  profile    | Profiles and benchmarks training loop"
        echo "  remove     | Removes conda environment"
        echo "  load       | Activates conda environment (use with source)"
        echo "  datasets   | Downloads datasets"
        echo "  help       | Print this help message"
        ;;
esac