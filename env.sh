#!/bin/bash
set -e


# Run command
case "$1" in
    # Create environments and install dependencies
    install)
        # Checkout to submodules' tracked version (also clones submodules if needed)
        git submodule update --init

        # Create conda environment for original implementation
        conda env create -f CaloDiffusion/conda.yml -y
        
        exit 0
        ;;

    # Load python venv
    load)
        set +e
        conda activate CaloDiffusion
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
        if [ "$1" != "help" ]; then
            echo "Unknown command '$1'"
        fi

        echo "Usage: ./env.sh COMMAND"
        echo "Commands:"
        echo "  install    | Installs submodules in python venv"
        echo "  remove     | Removes python venv"
        echo "  load       | Activates python venv (use with source)"
        echo "  datasets   | Downloads coco dataset, perform pose estimation with ViTPose and create KeypointsDataFrames"
        echo "  help       | Print this help message"
        ;;
esac