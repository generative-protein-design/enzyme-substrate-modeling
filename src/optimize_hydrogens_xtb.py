#!/usr/bin/env python3
"""
Keep heavy atoms fixed and optimize only the hydrogen atoms with xTB.

Example:
    python optimize_hydrogens_xtb.py lig_h.sdf

Outputs:
    constrain.inp
    lig_h.xtbopt.out
    lig_h_opt.sdf
    lig_h_opt.mol2

Requirements:
    xtb
    obabel
    rdkit
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Geometry import Point3D


BOHR_TO_ANGSTROM = 0.529177210903


def require_executable(name: str) -> None:
    """Fail early if an external command is missing."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def read_first_sdf_mol(path: Path) -> Chem.Mol:
    """Read the first molecule from an SDF file, preserving explicit H atoms."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)

    if len(supplier) == 0:
        raise ValueError(f"No molecules found in SDF file: {path}")

    mol = supplier[0]

    if mol is None:
        raise ValueError(f"Could not parse first molecule in SDF file: {path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"SDF has no 3D conformer/coordinates: {path}")

    return mol


def atom_indices_by_element(mol: Chem.Mol) -> tuple[list[int], list[int]]:
    """
    Return:
        heavy_atoms_1based, hydrogens_1based
    """
    heavy_atoms: list[int] = []
    hydrogens: list[int] = []

    for atom in mol.GetAtoms():
        idx_1based = atom.GetIdx() + 1
        atomic_num = atom.GetAtomicNum()

        if atomic_num == 1:
            hydrogens.append(idx_1based)
        elif atomic_num > 1:
            heavy_atoms.append(idx_1based)
        else:
            raise ValueError(
                f"Atom {idx_1based} has atomic number {atomic_num}; "
                "cannot decide whether to freeze it."
            )

    if not heavy_atoms:
        raise ValueError("No non-hydrogen atoms found to freeze.")

    if not hydrogens:
        raise ValueError(
            "No explicit hydrogens found. The SDF must contain explicit H atoms "
            "if the goal is to optimize only hydrogens."
        )

    return heavy_atoms, hydrogens


def compress_1based_indices(indices: Sequence[int]) -> list[str]:
    """
    Compress atom indices to xTB atom-list syntax.

    Example:
        [1, 2, 3, 7, 9, 10] -> ["1-3", "7", "9-10"]
    """
    values = sorted(set(indices))

    if not values:
        return []

    ranges: list[str] = []
    start = values[0]
    prev = values[0]

    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue

        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = value
        prev = value

    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ranges


def wrap_tokens(tokens: Sequence[str], max_chars: int = 90) -> list[str]:
    """Wrap comma-separated atom-list tokens over several atoms: lines."""
    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    for token in tokens:
        token_len = len(token) if not current else len(token) + 1

        if current and current_len + token_len > max_chars:
            lines.append(",".join(current))
            current = [token]
            current_len = len(token)
        else:
            current.append(token)
            current_len += token_len

    if current:
        lines.append(",".join(current))

    return lines


def write_xtb_fix_file(path: Path, fixed_atoms_1based: Sequence[int]) -> None:
    """
    Write xTB constraint file.

    Despite the conventional filename constrain.inp, this uses $fix,
    because $fix is exact Cartesian fixing in xTB.
    """
    tokens = compress_1based_indices(fixed_atoms_1based)
    wrapped_lines = wrap_tokens(tokens)

    with path.open("w") as handle:
        handle.write("$fix\n")
        for line in wrapped_lines:
            handle.write(f"   atoms: {line}\n")
        handle.write("$end\n")


def get_coords_from_rdkit_mol(mol: Chem.Mol) -> list[tuple[float, float, float]]:
    conf = mol.GetConformer()
    coords: list[tuple[float, float, float]] = []

    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append((float(p.x), float(p.y), float(p.z)))

    return coords


def read_coords_from_sdf(path: Path) -> list[tuple[float, float, float]]:
    mol = read_first_sdf_mol(path)
    return get_coords_from_rdkit_mol(mol)


def read_last_xyz_frame(path: Path) -> list[tuple[float, float, float]]:
    """
    Read the last frame from an XYZ/XMOL file.

    This works for both:
        xtbopt.xyz  single frame
        xtbopt.log  multi-frame optimization trajectory
    """
    lines = path.read_text().splitlines()
    frames: list[list[str]] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        try:
            natoms = int(line.split()[0])
        except ValueError:
            i += 1
            continue

        start = i + 2
        stop = start + natoms

        if stop <= len(lines):
            frames.append(lines[start:stop])

        i = stop

    if not frames:
        raise ValueError(f"No XYZ frames found in: {path}")

    coords: list[tuple[float, float, float]] = []

    for atom_line in frames[-1]:
        fields = atom_line.split()

        if len(fields) < 4:
            raise ValueError(f"Bad XYZ atom line in {path}: {atom_line!r}")

        x, y, z = map(float, fields[1:4])
        coords.append((x, y, z))

    return coords


def read_coords_from_turbomole_coord(path: Path) -> list[tuple[float, float, float]]:
    """
    Read a Turbomole coord file.

    xTB coord files use Bohr. Convert to Angstrom.
    """
    coords: list[tuple[float, float, float]] = []
    in_coord = False

    for line in path.read_text().splitlines():
        stripped = line.strip()

        if stripped.lower().startswith("$coord"):
            in_coord = True
            continue

        if in_coord and stripped.startswith("$"):
            break

        if not in_coord or not stripped:
            continue

        fields = stripped.split()

        if len(fields) < 4:
            continue

        x_bohr, y_bohr, z_bohr = map(float, fields[0:3])
        coords.append(
            (
                x_bohr * BOHR_TO_ANGSTROM,
                y_bohr * BOHR_TO_ANGSTROM,
                z_bohr * BOHR_TO_ANGSTROM,
            )
        )

    if not coords:
        raise ValueError(f"No coordinates found in Turbomole coord file: {path}")

    return coords


def read_optimized_coords(path: Path) -> list[tuple[float, float, float]]:
    suffix = path.suffix.lower()

    if suffix == ".sdf":
        return read_coords_from_sdf(path)

    if suffix in {".xyz", ".log"}:
        return read_last_xyz_frame(path)

    if suffix == ".coord" or path.name.endswith(".coord"):
        return read_coords_from_turbomole_coord(path)

    raise ValueError(f"Do not know how to read optimized coordinates from: {path}")


def find_xtb_optimized_geometry(work_dir: Path, namespace: str) -> Path:
    """
    Find xTB's optimized coordinate file.

    Different xTB builds/versions and input formats can produce slightly
    different names, so check namespace and non-namespace variants.
    """
    candidates = [
        work_dir / f"{namespace}.xtbopt.sdf",
        work_dir / "xtbopt.sdf",
        work_dir / f"{namespace}.xtbopt.xyz",
        work_dir / "xtbopt.xyz",
        work_dir / f"{namespace}.xtbopt.coord",
        work_dir / "xtbopt.coord",
        work_dir / f"{namespace}.xtbopt.log",
        work_dir / "xtbopt.log",
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path

    found = sorted(p.name for p in work_dir.glob("*"))
    raise FileNotFoundError(
        "Could not find xTB optimized geometry. Checked:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\n\nFiles present in working directory:\n"
        + "\n".join(f"  {name}" for name in found)
    )


def replace_mol_coords(
    mol: Chem.Mol,
    coords_angstrom: Sequence[tuple[float, float, float]],
    name: str | None = None,
) -> Chem.Mol:
    """Return a copy of mol with its conformer coordinates replaced."""
    if len(coords_angstrom) != mol.GetNumAtoms():
        raise ValueError(
            f"Coordinate count mismatch: got {len(coords_angstrom)} coordinates "
            f"for {mol.GetNumAtoms()} atoms."
        )

    out = Chem.Mol(mol)

    if out.GetNumConformers() == 0:
        conf = Chem.Conformer(out.GetNumAtoms())
        out.AddConformer(conf, assignId=True)

    conf = out.GetConformer()

    for i, (x, y, z) in enumerate(coords_angstrom):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    if name is not None:
        out.SetProp("_Name", name)

    return out


def write_sdf(mol: Chem.Mol, path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write SDF: {path}")


def run_obabel_sdf_to_mol2(input_sdf: Path, output_mol2: Path) -> None:
    cmd = [
        "obabel",
        "-isdf",
        str(input_sdf),
        "-omol2",
        "-O",
        str(output_mol2),
    ]

    subprocess.run(cmd, check=True, text=True)

    if not output_mol2.exists() or output_mol2.stat().st_size == 0:
        raise RuntimeError(f"Open Babel did not create MOL2 output: {output_mol2}")


def read_mol2_atom_names(path: Path) -> list[str]:
    names: list[str] = []
    in_atom_section = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()

        if line.upper() == "@<TRIPOS>ATOM":
            in_atom_section = True
            continue

        if line.startswith("@<TRIPOS>"):
            if in_atom_section:
                break
            continue

        if not in_atom_section or not line:
            continue

        fields = raw_line.split()
        if len(fields) < 2:
            continue

        names.append(fields[1])

    if not names:
        raise ValueError(f"No atoms found in MOL2 ATOM section: {path}")

    return names


def restore_mol2_atom_and_residue_names(
    target_mol2: Path,
    reference_mol2: Path,
    residue_name: str = "LIG1",
) -> None:
    ref_atom_names = read_mol2_atom_names(reference_mol2)
    target_lines = target_mol2.read_text().splitlines()

    out_lines: list[str] = []
    in_atom_section = False
    atom_idx = 0

    for raw_line in target_lines:
        line = raw_line.strip()

        if line.upper() == "@<TRIPOS>ATOM":
            in_atom_section = True
            out_lines.append(raw_line)
            continue

        if line.startswith("@<TRIPOS>"):
            in_atom_section = False
            out_lines.append(raw_line)
            continue

        if not in_atom_section or not line:
            out_lines.append(raw_line)
            continue

        fields = raw_line.split()
        if len(fields) < 8:
            raise ValueError(f"Unexpected MOL2 atom line format in {target_mol2}: {raw_line!r}")

        if atom_idx >= len(ref_atom_names):
            raise ValueError(
                f"Target MOL2 has more atoms than reference names in {reference_mol2}"
            )

        fields[1] = ref_atom_names[atom_idx]
        fields[7] = residue_name
        atom_idx += 1

        formatted = (
            f"{fields[0]:>7} {fields[1]:<8} {fields[2]:>10} {fields[3]:>10} "
            f"{fields[4]:>10} {fields[5]:<7} {fields[6]:>4}  {fields[7]:<8}"
        )

        if len(fields) > 8:
            formatted += f" {fields[8]:>10}"

        if len(fields) > 9:
            formatted += " " + " ".join(fields[9:])

        out_lines.append(formatted)

    if atom_idx == 0:
        raise ValueError(f"No target atoms found in MOL2 ATOM section: {target_mol2}")

    if atom_idx != len(ref_atom_names):
        raise ValueError(
            f"Atom count mismatch for name restore: target has {atom_idx}, "
            f"reference has {len(ref_atom_names)}"
        )

    target_mol2.write_text("\n".join(out_lines) + "\n")


def rmsd_for_atoms(
    coords_a: Sequence[tuple[float, float, float]],
    coords_b: Sequence[tuple[float, float, float]],
    atoms_1based: Iterable[int],
) -> float:
    sq = 0.0
    n = 0

    for idx_1based in atoms_1based:
        i = idx_1based - 1

        ax, ay, az = coords_a[i]
        bx, by, bz = coords_b[i]

        sq += (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        n += 1

    if n == 0:
        return 0.0

    return math.sqrt(sq / n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run xTB with all non-hydrogen atoms fixed and hydrogens optimized. "
            "Writes optimized SDF and MOL2."
        )
    )

    parser.add_argument(
        "input_sdf",
        nargs="?",
        default="lig_h.sdf",
        type=Path,
        help="Input SDF with explicit hydrogens and 3D coordinates. Default: lig_h.sdf",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total molecular charge. Default: 0",
    )
    parser.add_argument(
        "--uhf",
        type=int,
        default=0,
        help="Number of unpaired electrons. Default: 0",
    )
    parser.add_argument(
        "--solvent",
        default="water",
        help="ALPB solvent. Default: water",
    )
    parser.add_argument(
        "--alpb-state",
        default=None,
        choices=["reference", "bar1M", "bar1mol", "gsolv"],
        help=(
            "Optional ALPB reference state. xTB documents reference and bar1M; "
            "omit to use xTB's default."
        ),
    )
    parser.add_argument(
        "--gfn",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="GFN-xTB Hamiltonian. Default: 2",
    )
    parser.add_argument(
        "--opt-level",
        default="normal",
        choices=["crude", "sloppy", "loose", "normal", "tight", "verytight", "extreme"],
        help="xTB optimization level. Default: normal",
    )
    parser.add_argument(
        "--prefix",
        default="lig_h",
        help=(
            "Prefix for xTB namespace and output files. "
            "Default: lig_h, giving lig_h_opt.sdf and lig_h_opt.mol2"
        ),
    )
    parser.add_argument(
        "--verbose-xtb",
        action="store_true",
        help="Use xTB --verbose instead of --silent.",
    )
    parser.add_argument(
        "--keep-junk",
        action="store_true",
        help="Keep auxiliary xTB files such as charges, wbo, and xtbrestart.",
    )

    args = parser.parse_args()

    input_sdf = args.input_sdf.resolve()

    if not input_sdf.exists():
        raise FileNotFoundError(f"Input SDF does not exist: {input_sdf}")

    require_executable("xtb")
    require_executable("obabel")

    work_dir = input_sdf.parent
    namespace = args.prefix

    constrain_file = work_dir / "constrain.inp"
    xtb_output_log = work_dir / f"{namespace}.xtbopt.out"

    output_sdf = work_dir / f"{namespace}_opt.sdf"
    output_mol2 = work_dir / f"{namespace}_opt.mol2"

    input_mol = read_first_sdf_mol(input_sdf)
    input_coords = get_coords_from_rdkit_mol(input_mol)

    fixed_atoms, movable_hydrogens = atom_indices_by_element(input_mol)

    write_xtb_fix_file(constrain_file, fixed_atoms)

    print(f"Wrote: {constrain_file}")
    print(f"Frozen non-H atoms: {len(fixed_atoms)}")
    print(f"Movable H atoms:     {len(movable_hydrogens)}")

    cmd = [
        "xtb",
        str(input_sdf),
        "--gfn",
        str(args.gfn),
        "--chrg",
        str(args.charge),
        "--uhf",
        str(args.uhf),
        "--alpb",
        args.solvent,
    ]

    if args.alpb_state is not None:
        cmd.append(args.alpb_state)

    cmd.extend(
        [
            "--opt",
            args.opt_level,
            "--input",
            str(constrain_file),
            "--namespace",
            namespace,
        ]
    )

    if args.verbose_xtb:
        cmd.append("--verbose")
    else:
        cmd.append("--silent")

    print("Running xTB:")
    print("  " + " ".join(cmd))
    print(f"xTB output log: {xtb_output_log}")

    with xtb_output_log.open("w") as handle:
        subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    xtb_geom = find_xtb_optimized_geometry(work_dir, namespace)
    print(f"Found xTB optimized coordinates: {xtb_geom}")

    opt_coords = read_optimized_coords(xtb_geom)

    heavy_rmsd = rmsd_for_atoms(input_coords, opt_coords, fixed_atoms)
    print(f"Heavy-atom RMSD before/after: {heavy_rmsd:.8f} Å")

    if heavy_rmsd > 1.0e-4:
        print(
            "WARNING: heavy atoms moved by more than 1e-4 Å. "
            "Check constrain.inp and the xTB output log.",
            file=sys.stderr,
        )

    opt_mol = replace_mol_coords(
        input_mol,
        opt_coords,
        name=f"{namespace}_opt",
    )

    write_sdf(opt_mol, output_sdf)
    print(f"Wrote: {output_sdf}")

    run_obabel_sdf_to_mol2(output_sdf, output_mol2)
    reference_mol2 = work_dir / "lig_h.mol2"
    if not reference_mol2.exists():
        raise FileNotFoundError(
            f"Reference MOL2 for atom names was not found: {reference_mol2}"
        )
    restore_mol2_atom_and_residue_names(output_mol2, reference_mol2, residue_name="LIG1")
    print(f"Wrote: {output_mol2}")

    if not args.keep_junk:
        junk_names = [
            f"{namespace}.charges",
            "charges",
            f"{namespace}.wbo",
            "wbo",
            f"{namespace}.xtbrestart",
            "xtbrestart",
            f"{namespace}.xtbtopo.sdf",
            "xtbtopo.sdf",
        ]

        for name in junk_names:
            path = work_dir / name
            if path.exists():
                path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
