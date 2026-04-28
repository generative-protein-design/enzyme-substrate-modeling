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
        --amber_params=*)
            AMBER_PARAMS="${ARG#*=}"
            ;;
        --residue_number=*)
            RES_NUM="${ARG#*=}"
            ;;
        *)
            echo "Unknown argument: $ARG"
            exit 1
            ;;
    esac
done

BASE_DIR=`pwd`
MODEL=$(basename "$INPUT_MODEL" .pdb)
mkdir -p $OUTPUT_FOLDER/$MODEL


pixi run -e analysis python src/add_hydrogens_obabel.py $INPUT_MODEL $OUTPUT_FOLDER/$MODEL
#pixi run -e analysis obabel $OUTPUT_FOLDER/$MODEL/lig_pymol.pdb -O $OUTPUT_FOLDER/$MODEL/lig_h.pdb -p 7.4
#pixi run -e analysis obabel $OUTPUT_FOLDER/$MODEL/lig_pymol.mol2 -O $OUTPUT_FOLDER/$MODEL/lig_h.mol2 -p 7.4

# Generate Amber prmtop and inpcrd files with tleap. Run this step for every unique sequence.
#https://ambermd.org/tutorials/basic/tutorial4b/

cp $BASE_DIR/templates/leap2.in $OUTPUT_FOLDER/$MODEL
sed -i "s/__RES_NUM__/${RES_NUM}/g" $OUTPUT_FOLDER/$MODEL/leap2.in

cd $OUTPUT_FOLDER/$MODEL
grep -v -e LIG -e CONECT -e '^[[:space:]]*$' ${INPUT_MODEL} > protein.pdb
ln -sf $AMBER_PARAMS/lig.frcmod .
ln -sf $AMBER_PARAMS/lig.lib .

pixi run -e analysis tleap -f leap2.in
pixi run -e analysis python $BASE_DIR/src/run_openmm.py

pixi run -e analysis python $BASE_DIR/src/molfile_to_params.py \
	-p LIG \
	-n LIG \
	--long-names \
	--clobber \
	--keep-names \
	lig_h.mol2

pixi run -e analysis python $BASE_DIR/src/pyrosetta_interface_delta.py

cd - > /dev/null


