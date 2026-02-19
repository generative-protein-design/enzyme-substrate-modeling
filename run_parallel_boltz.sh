#!/bin/bash

#SBATCH --account m906
#SBATCH --constraint gpu&hbm80g
#SBATCH --qos premium
#SBATCH --time 23:00:00
#SBATCH --nodes 20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4

export SLURM_CPU_BIND="cores"

set -e
umask 000


export CONFIG_NAME=${1:-config}
export OUTPUT_FOLDER=$(pixi run --as-is -e rf python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.work_dir).name)")

echo output folder: $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER

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


pixi run --as-is -e boltz python prepare_boltz_input_nyl12.py +site=perlmutter colabfold.custom_template_path=\$CIF_FOLDER --config-name=$CONFIG_NAME


run_task msas_convert <<'CMD'
srun driver.sh ${OUTPUT_FOLDER}/2_boltz/commands_msas_convert.sh
CMD

run_task boltz <<'CMD'
srun driver.sh ${OUTPUT_FOLDER}/2_boltz/commands_boltz2.sh
CMD


export COLABFOLD_TEMPLATE_FOLDER=$(pixi run --as-is -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(cfg.colabfold.custom_template_path)")

#colabfold_local
run_task colabfold_local <<'CMD'
srun prepare_cif.sh $COLABFOLD_TEMPLATE_FOLDER  &&
srun driver.sh ${OUTPUT_FOLDER}/2b_colabfold/commands_colabfold.sh
CMD



export BASE_DIR=`pwd`
export RELAXATION_OUTPUT_FOLDER=$(pixi run --as-is -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.relaxation.output_dir).name)")
export BOLTZ_OUTPUT_FOLDER=$(pixi run --as-is -e analysis python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.boltz.output_dir).name)")
export INPUT_MODEL=$(find $BASE_DIR/$OUTPUT_FOLDER/$BOLTZ_OUTPUT_FOLDER -type f -path "*/boltz*/predictions/*/*.pdb" | head -n 1)

run_task amber_params <<'CMD'
pixi run --as-is -e analysis bash src/compute_amber_params.sh --input_model=$INPUT_MODEL --output_folder=$BASE_DIR/$OUTPUT_FOLDER/$RELAXATION_OUTPUT_FOLDER
CMD

run_task relaxation <<'CMD'
bash src/prepare_relaxation_commands.sh --command=$BASE_DIR/src/run_relaxation.sh --input_folder=$BASE_DIR/$OUTPUT_FOLDER/$BOLTZ_OUTPUT_FOLDER --output_folder=$BASE_DIR/$OUTPUT_FOLDER/$RELAXATION_OUTPUT_FOLDER &&
srun --kill-on-bad-exit=0 driver.sh ${OUTPUT_FOLDER}/${RELAXATION_OUTPUT_FOLDER}/commands_relaxation.sh || true
CMD

run_task filtering <<'CMD'
pixi run --as-is -e analysis python analyze_colabfold_models.py +site=perlmutter --config-name=$CONFIG_NAME &&
pixi run --as-is -e analysis python analyze_boltz_models.py +site=perlmutter --config-name=$CONFIG_NAME
CMD
