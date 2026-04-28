#!/usr/bin/env bash

set -e

export NTASKS_CPU=16
export NTASKS_GPU=2

export BASE_DIR=`pwd`

export CONFIG_NAME=${1:-config}

export OUTPUT_FOLDER=$(pixi run --as-is -e boltz python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.work_dir).name)")

run_task() {
    local DONE="$1"
    if [[ -f ${OUTPUT_FOLDER}/${DONE}.done ]]; then
        echo "[SKIP] $DONE"
        return
    fi
    echo "[BEGIN] $DONE"
    local start_time=$SECONDS
    cat | bash
    local duration=$(( SECONDS - start_time ))
    echo "[END] $DONE (took ${duration}s)"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "Completed on: $timestamp in ${duration}s" > "${OUTPUT_FOLDER}/${DONE}.done"
}


#boltz2
pixi run --as-is -e boltz python prepare_boltz_input.py +site=aster --config-name=$CONFIG_NAME

run_task boltz <<'CMD'
parallel --halt soon,fail=1 -j $NTASKS_GPU --ungroup CUDA_VISIBLE_DEVICES='$(({%} - 1))' bash -c "{}" :::: ${OUTPUT_FOLDER}/0_boltz/commands_boltz2.sh
CMD

pixi run --as-is -e boltz python prepare_relaxation_input.py +site=aster --config-name=$CONFIG_NAME

export RELAXATION_OUTPUT_FOLDER=$(pixi run --as-is -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.relaxation.output_dir).name)")


run_task amber_params <<'CMD'
bash ${OUTPUT_FOLDER}/${RELAXATION_OUTPUT_FOLDER}/commands_amber.sh
CMD

run_task relaxation <<'CMD'
#bash src/prepare_relaxation_commands.sh --command=$BASE_DIR/src/run_relaxation.sh --input_folder=$BASE_DIR/$OUTPUT_FOLDER/$BOLTZ_OUTPUT_FOLDER --output_folder=$BASE_DIR/$OUTPUT_FOLDER/$RELAXATION_OUTPUT_FOLDER &&
parallel -j $NTASKS_CPU --ungroup bash -c "{}" :::: ${OUTPUT_FOLDER}/${RELAXATION_OUTPUT_FOLDER}/commands_relaxation.sh || true
CMD

run_task filtering <<'CMD'
pixi run --as-is -e analysis python analyze_boltz_models.py +site=aster --config-name=$CONFIG_NAME
CMD


#chmod og+rwX -R .
