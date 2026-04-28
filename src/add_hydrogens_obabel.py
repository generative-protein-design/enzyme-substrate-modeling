#!/usr/bin/env python3
"""
Extract a ligand residue from a PDB and add hydrogens using OpenBabel at pH 7.4.

Usage:
    python add_hydrogens_obabel.py input.pdb output_folder [ligand_resname]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def _pdb_element(line: str) -> str:
    element = line[76:78].strip()
    if element:
        return element.upper()
    name = line[12:16].strip()
    return (name[0] if name else "X").upper()


def extract_ligand_pdb(input_pdb: Path, output_pdb: Path, ligand_resname: str) -> None:
    ligand_lines = []
    selected_residue = None
    with input_pdb.open("r") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[17:20].strip() != ligand_resname:
                continue

            key = (line[21], int(line[22:26]), line[26])
            if selected_residue is None:
                selected_residue = key
            if key != selected_residue:
                continue
            ligand_lines.append(line)

    if not ligand_lines:
        raise ValueError(f"No atoms found for residue name '{ligand_resname}' in input model")

    with output_pdb.open("w") as handle:
        handle.writelines(ligand_lines)
        handle.write("END\n")


def run_obabel_protonate(input_pdb: Path, output_file: Path, out_format: str) -> None:
    obabel_exe = shutil.which("obabel")
    if obabel_exe is None:
        raise RuntimeError("OpenBabel executable 'obabel' was not found in PATH")

    cmd = [
        obabel_exe,
        "-ipdb",
        str(input_pdb),
        f"-o{out_format}",
        "-O",
        str(output_file),
        "-p",
        "7.4",
    ]
    subprocess.run(cmd, check=True)


def run_obabel_convert(input_pdb: Path, output_file: Path, out_format: str) -> None:
    obabel_exe = shutil.which("obabel")
    if obabel_exe is None:
        raise RuntimeError("OpenBabel executable 'obabel' was not found in PATH")

    cmd = [
        obabel_exe,
        "-ipdb",
        str(input_pdb),
        f"-o{out_format}",
        "-O",
        str(output_file),
    ]
    subprocess.run(cmd, check=True)


def infer_net_charge_from_obabel(input_pdb: Path) -> int:
    obabel_exe = shutil.which("obabel")
    if obabel_exe is None:
        raise RuntimeError("OpenBabel executable 'obabel' was not found in PATH")

    cmd = [
        obabel_exe,
        "-ipdb",
        str(input_pdb),
        "-oinchi",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    inchi = ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("InChI="):
            inchi = line
            break

    if not inchi:
        raise RuntimeError("Unable to parse InChI from OpenBabel output for charge inference")

    net_charge = 0

    q_match = re.search(r"/q([+-]?\d+(?:;[+-]?\d+)*)", inchi)
    if q_match:
        q_values = q_match.group(1).split(";")
        net_charge += sum(int(v) for v in q_values)

    p_match = re.search(r"/p([+-]?\d+)", inchi)
    if p_match:
        net_charge += int(p_match.group(1))

    return net_charge


def _parse_ligand_atoms(
    pdb_path: Path,
    ligand_resname: str,
    lig_only: bool = True,
) -> list[str]:
    atoms: list[str] = []
    with pdb_path.open("r") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if lig_only and line[17:20].strip() != ligand_resname:
                continue
            atoms.append(line.rstrip("\n"))
    return atoms


def _format_h_line(
    serial: int,
    atom_name: str,
    template_line: str,
    x: float,
    y: float,
    z: float,
) -> str:
    return (
        f"HETATM{serial:5d} {atom_name:>4}{template_line[16:30]}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}{template_line[54:76]}{'H':>2}{template_line[78:]}"
    )


def build_coordinate_preserving_ligand_pdb(
    raw_pdb: Path,
    protonated_tmp_pdb: Path,
    output_pdb: Path,
    ligand_resname: str,
) -> None:
    raw_atoms = _parse_ligand_atoms(raw_pdb, ligand_resname=ligand_resname)
    if not raw_atoms:
        raise ValueError("No ligand atoms found in raw ligand PDB")

    heavy_atoms = [line for line in raw_atoms if _pdb_element(line) != "H"]

    protonated_atoms = _parse_ligand_atoms(
        protonated_tmp_pdb,
        ligand_resname=ligand_resname,
        lig_only=False,
    )
    hydrogens = [line for line in protonated_atoms if _pdb_element(line) == "H"]

    serial = 1
    out_lines: list[str] = []
    for line in heavy_atoms:
        out_lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
        serial += 1

    template = heavy_atoms[0]
    for i, hline in enumerate(hydrogens, start=1):
        x = float(hline[30:38])
        y = float(hline[38:46])
        z = float(hline[46:54])
        hname = f"H{i}"[-4:]
        out_lines.append(_format_h_line(serial, hname, template, x, y, z))
        serial += 1

    out_lines.append("END")
    output_pdb.write_text("\n".join(out_lines) + "\n")


def add_hydrogens_to_ligand(pdb_file: Path, output_dir: Path, ligand_resname: str) -> None:
    ligand_raw = output_dir / "lig_raw.pdb"
    ligand_h_tmp = output_dir / "lig_h_tmp.pdb"
    output_pdb = output_dir / "lig_h.pdb"
    output_mol2 = output_dir / "lig_h.mol2"
    output_sdf = output_dir / "lig_h.sdf"
    output_charge = output_dir / "lig_charge.txt"

    extract_ligand_pdb(pdb_file, ligand_raw, ligand_resname=ligand_resname)
    run_obabel_protonate(ligand_raw, ligand_h_tmp, "pdb")
    build_coordinate_preserving_ligand_pdb(
        ligand_raw,
        ligand_h_tmp,
        output_pdb,
        ligand_resname=ligand_resname,
    )
    run_obabel_convert(output_pdb, output_mol2, "mol2")
    run_obabel_convert(output_pdb, output_sdf, "sdf")
    net_charge = infer_net_charge_from_obabel(output_pdb)
    output_charge.write_text(f"{net_charge}\n")

    print(
        f"Ligand with hydrogens written to: {output_pdb}, {output_mol2}, and {output_sdf}; "
        f"inferred net charge={net_charge} (saved in {output_charge})"
    )


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python add_hydrogens_obabel.py input.pdb output_folder [ligand_resname]")
        sys.exit(1)

    pdb_file = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    ligand_resname = sys.argv[3] if len(sys.argv) == 4 else "LIG"
    output_folder.mkdir(parents=True, exist_ok=True)
    add_hydrogens_to_ligand(pdb_file, output_folder, ligand_resname=ligand_resname)
