export CONFIG_NAME=${1:-config}
export OUTPUT_FOLDER=$(pixi run -e boltz python -c "from omegaconf import OmegaConf; from pathlib import Path; cfg = OmegaConf.load('config/${CONFIG_NAME}.yaml'); print(Path(cfg.work_dir).name)")
LOGFILE="${OUTPUT_FOLDER}/output_$(date '+%Y-%m-%d_%H-%M-%S').log"

echo starting protein engineering
echo output folder: $OUTPUT_FOLDER
echo logs:  $LOGFILE


mkdir -p $OUTPUT_FOLDER
setsid ./run_all_aster.sh $CONFIG_NAME &> "$LOGFILE" &



