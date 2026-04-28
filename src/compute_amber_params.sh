#!/bin/bash

set -e

for ARG in "$@"; do
    case $ARG in
        --input_model=*)
            INPUT_MODEL="${ARG#*=}"
            ;;
        --output_folder=*)
            OUTPUT_FOLDER="${ARG#*=}"
            ;;
        --charge=*)
            CHARGE="${ARG#*=}"
            ;;
        *)
            echo "Unknown argument: $ARG"
            exit 1
            ;;
    esac
done


BASE_DIR=`pwd`

MODEL=$(basename "$INPUT_MODEL" .pdb)


echo computing Amber parameters using $INPUT_MODEL ...

# Generate Amber parameters with antechamber.
mkdir -p $OUTPUT_FOLDER
cp $BASE_DIR/config/leap.in $OUTPUT_FOLDER
cd $OUTPUT_FOLDER
pixi run -e analysis python $BASE_DIR/src/add_hydrogens_obabel.py $INPUT_MODEL .
pixi run -e analysis $BASE_DIR/src/xtb_optimize_hydrogens.py lig_h.sdf --verbose
pixi run -e analysis antechamber -i lig_h_opt.sdf -fi sdf -o lig.mol2 -fo mol2 -c bcc -s 2 -nc $CHARGE -ek "maxcyc=0" # JMP: Don't optimize the geometry.
pixi run -e analysis parmchk2 -i lig.mol2 -f mol2 -o lig.frcmod
pixi run -e analysis tleap -f leap.in
cd - > /dev/null

echo Amber parameters are saved in $OUTPUT_FOLDER




