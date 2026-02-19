#!/usr/bin/env python3
"""
Add hydrogens to a ligand (resn LIG) using the PyMOL API.

Usage:
    python add_hydrogens_pymol.py input.pdb
"""

import sys
from pathlib import Path

from pymol import cmd

def add_hydrogens_to_ligand(pdb_file, output_dir):
    # Clear PyMOL session
    cmd.reinitialize()

    # Load the structure
    cmd.load(pdb_file, "complex")

    # Select the ligand by residue name
    ligand_sel = "resn LIG"

    # Check that the ligand exists
    if cmd.count_atoms(ligand_sel) == 0:
        raise ValueError(f"No atoms found for selection '{ligand_sel}'")

    # Extract ligand to its own object
    cmd.create("ligand", ligand_sel)

    # Add hydrogens to the ligand
#    cmd.h_add("ligand")

    pdb_path = Path(pdb_file)
    output_pdb_file = output_dir / "lig_h_pymol.pdb"
    output_mol_file = output_dir / "lig_h_pymol.mol2"

    cmd.save(output_pdb_file, "ligand")
    cmd.save(output_mol_file, "ligand")

#    print(f"Ligand with hydrogens written to: {output_pdb_file}")
    print(f"Ligand with hydrogens written to: {output_pdb_file} and {output_mol_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_hydrogens_pymol.py input.pdb output_folder")
        sys.exit(1)

    pdb_file = sys.argv[1]
    output_folder = sys.argv[2]
    dir_path = Path(output_folder)
    dir_path.mkdir(parents=True, exist_ok=True)
    add_hydrogens_to_ligand(pdb_file, dir_path)

