#!/bin/bash

set -e

for ARG in "$@"; do
    case $ARG in
        --command=*)
            COMMAND="${ARG#*=}"
            ;;
        --input_folder=*)
            INPUT_FOLDER="${ARG#*=}"
            ;;
        --output_folder=*)
            OUTPUT_FOLDER="${ARG#*=}"
            ;;
        *)
            echo "Unknown argument: $ARG"
            exit 1
            ;;
    esac
done

INPUT_MODELS=$(find $INPUT_FOLDER -type f -path "*/boltz*/predictions/*/*.pdb")

echo -n "" > $OUTPUT_FOLDER/commands_relaxation.sh

for model in $INPUT_MODELS; do
  command="bash $COMMAND --input_model=$model --output_folder=$OUTPUT_FOLDER"
  echo $command >> $OUTPUT_FOLDER/commands_relaxation.sh
done



