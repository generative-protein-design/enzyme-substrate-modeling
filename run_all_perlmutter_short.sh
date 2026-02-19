#!/bin/bash

set -e

jobid=$(sbatch --time 4:00:00 --parsable run_single_before_boltz.sh "$@")
sbatch --nodes 2 --time 2:00:00 --dependency=afterok:$jobid run_parallel_boltz.sh "$@"