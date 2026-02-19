#!/bin/bash

set -e

jobid=$(sbatch --parsable run_single_before_boltz.sh "$@")
sbatch --dependency=afterok:$jobid run_parallel_boltz.sh "$@"