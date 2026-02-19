#!/bin/bash

#SBATCH --account m906
#SBATCH --constraint gpu&hbm80g
#SBATCH --qos regular
#SBATCH --time 8:00:00
#SBATCH --nodes 1
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


#rf-diffusion
run_task rf_diffusion <<'CMD'
pixi run --as-is -e rf python 0_run_diffusion_nyl12.py +site=perlmutter --config-name=$CONFIG_NAME &&
srun driver.sh ${OUTPUT_FOLDER}/0_diffusion/commands_diffusion.sh
CMD

#ligand-mpnn
run_task ligand_mpnn <<'CMD'
pixi run --as-is -e ligand python 1_ligandmpnn_nyl12.py +site=perlmutter --config-name=$CONFIG_NAME &&
srun driver.sh ${OUTPUT_FOLDER}/1_ligandmpnn/commands_mpnn.sh
CMD

#boltz2
pixi run --as-is -e boltz python prepare_boltz_input_nyl12.py +site=perlmutter colabfold.custom_template_path=\$CIF_FOLDER --config-name=$CONFIG_NAME

run_task colabfold_search <<'CMD'
bash ${OUTPUT_FOLDER}/2_boltz/commands_colabfold_search.sh
CMD

