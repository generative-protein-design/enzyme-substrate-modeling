#!/usr/bin/env python3

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def normalize_msa_sequence(seq: str):
    return "".join(ch for ch in seq if ch.isupper()).replace("-", "")


def validate_a3m_query_first(a3m_path: Path, expected_query_sequence: str):
    lines = [ln.strip() for ln in a3m_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise ValueError(f"Invalid A3M format: {a3m_path}")

    first_header = lines[0]
    first_seq = lines[1]
    first_norm = normalize_msa_sequence(first_seq)
    query_norm = normalize_msa_sequence(expected_query_sequence)

    if first_norm != query_norm:
        raise ValueError(
            "Local A3M first sequence does not match query sequence. "
            "Boltz may ignore malformed MSA usage in this situation. "
            f"File={a3m_path}, first_header={first_header}"
        )

    return {
        "a3m_path": str(a3m_path),
        "first_header": first_header,
        "first_sequence_length": len(first_seq),
        "normalized_query_length": len(query_norm),
    }


def build_boltz_named_mol_from_smiles(smiles: str):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse ligand SMILES for scissile atom inference")

    mol = AllChem.AddHs(mol)
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"SMILES generated an atom name longer than 4 chars: {atom_name}"
            )
        atom.SetProp("name", atom_name)

    return mol


def infer_amide_nitrogen_idx(mol, carbon_idx: int, candidate_indices=None):
    c_atom = mol.GetAtomWithIdx(carbon_idx)
    cands = []
    for bond in c_atom.GetBonds():
        nbr = bond.GetOtherAtom(c_atom)
        if nbr.GetAtomicNum() != 7:
            continue
        if bond.GetBondTypeAsDouble() < 1.0:
            continue
        nbr_idx = nbr.GetIdx()
        if candidate_indices is not None and nbr_idx not in candidate_indices:
            continue
        cands.append(nbr_idx)
    if not cands:
        return None
    return min(cands)


def infer_scissile_atom_names(
    ligand_smiles: str,
    amide_smarts: str,
    carbon_mapnum: int,
    oxygen_mapnum: int,
    nitrogen_mapnum: int,
    match_index: int,
):
    mol = build_boltz_named_mol_from_smiles(ligand_smiles)
    patt = Chem.MolFromSmarts(amide_smarts)
    if patt is None:
        raise ValueError(f"Could not parse scissile SMARTS: {amide_smarts}")

    mapnum_to_query_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in patt.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }

    if carbon_mapnum not in mapnum_to_query_idx:
        raise ValueError(
            f"Carbon map number {carbon_mapnum} not found in SMARTS: {amide_smarts}"
        )
    if oxygen_mapnum not in mapnum_to_query_idx:
        raise ValueError(
            f"Oxygen map number {oxygen_mapnum} not found in SMARTS: {amide_smarts}"
        )

    use_mapped_n = nitrogen_mapnum in mapnum_to_query_idx

    matches = list(mol.GetSubstructMatches(patt, uniquify=True))
    if not matches:
        raise ValueError(
            "No SMARTS matches were found in ligand for scissile atom inference"
        )

    all_matches = []
    for i, match in enumerate(matches):
        c_atom_idx = match[mapnum_to_query_idx[carbon_mapnum]]
        o_atom_idx = match[mapnum_to_query_idx[oxygen_mapnum]]
        if use_mapped_n:
            n_atom_idx = match[mapnum_to_query_idx[nitrogen_mapnum]]
        else:
            n_atom_idx = infer_amide_nitrogen_idx(mol, c_atom_idx)
        c_atom = mol.GetAtomWithIdx(c_atom_idx)
        o_atom = mol.GetAtomWithIdx(o_atom_idx)
        n_atom = mol.GetAtomWithIdx(n_atom_idx) if n_atom_idx is not None else None

        all_matches.append(
            {
                "match_index": i,
                "carbon_atom_idx": int(c_atom_idx),
                "oxygen_atom_idx": int(o_atom_idx),
                "carbon_atomic_num": int(c_atom.GetAtomicNum()),
                "oxygen_atomic_num": int(o_atom.GetAtomicNum()),
                "nitrogen_atom_idx": int(n_atom_idx) if n_atom_idx is not None else None,
                "nitrogen_atomic_num": int(n_atom.GetAtomicNum()) if n_atom is not None else None,
                "carbon_atom_name": c_atom.GetProp("name"),
                "oxygen_atom_name": o_atom.GetProp("name"),
                "nitrogen_atom_name": n_atom.GetProp("name") if n_atom is not None else None,
            }
        )

    if match_index < 0 or match_index >= len(all_matches):
        raise ValueError(
            f"scissile-match-index {match_index} out of range for {len(all_matches)} SMARTS matches"
        )

    chosen = all_matches[match_index]
    if (
        chosen["carbon_atomic_num"] != 6
        or chosen["oxygen_atomic_num"] != 8
        or chosen["nitrogen_atomic_num"] != 7
    ):
        raise ValueError(
            "Selected SMARTS mapping did not resolve to C/O/N atoms for the scissile amide"
        )

    return (
        chosen["carbon_atom_name"],
        chosen["oxygen_atom_name"],
        chosen["nitrogen_atom_name"],
        all_matches,
    )


def infer_scissile_atom_names_from_context_smarts(
    ligand_smiles: str,
    context_smarts: str,
    match_index: int,
    require_unique: bool = False,
):
    mol = build_boltz_named_mol_from_smiles(ligand_smiles)
    patt = Chem.MolFromSmarts(context_smarts)
    if patt is None:
        raise ValueError(f"Could not parse scissile context SMARTS: {context_smarts}")

    matches = list(mol.GetSubstructMatches(patt, uniquify=True))
    if not matches:
        raise ValueError(
            "No context SMARTS matches were found in ligand for scissile atom inference"
        )

    mapnum_to_query_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in patt.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    use_mapped_atoms = 1 in mapnum_to_query_idx and 2 in mapnum_to_query_idx
    use_mapped_n = 3 in mapnum_to_query_idx

    all_matches = []
    for i, match in enumerate(matches):
        if use_mapped_atoms:
            carbon_idx = match[mapnum_to_query_idx[1]]
            oxygen_idx = match[mapnum_to_query_idx[2]]
            if use_mapped_n:
                nitrogen_idx = match[mapnum_to_query_idx[3]]
            else:
                nitrogen_idx = infer_amide_nitrogen_idx(mol, carbon_idx, set(match))
        else:
            matched_atom_indices = set(match)
            amide_triplets = []

            for carbon_idx in matched_atom_indices:
                c_atom = mol.GetAtomWithIdx(carbon_idx)
                if c_atom.GetAtomicNum() != 6:
                    continue

                oxygen_neighbors = []
                nitrogen_neighbors = []
                for bond in c_atom.GetBonds():
                    nbr = bond.GetOtherAtom(c_atom)
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx not in matched_atom_indices:
                        continue

                    btype = bond.GetBondTypeAsDouble()
                    if nbr.GetAtomicNum() == 8 and abs(btype - 2.0) < 0.2:
                        oxygen_neighbors.append(nbr_idx)
                    elif nbr.GetAtomicNum() == 7 and btype >= 1.0:
                        nitrogen_neighbors.append(nbr_idx)

                if nitrogen_neighbors and oxygen_neighbors:
                    for oi in oxygen_neighbors:
                        for ni in nitrogen_neighbors:
                            amide_triplets.append((carbon_idx, oi, ni))

            if len(amide_triplets) != 1:
                raise ValueError(
                    "Context SMARTS match did not resolve to exactly one amide C/O/N triplet "
                    f"(match_index={i}, triplet_count={len(amide_triplets)})."
                )

            carbon_idx, oxygen_idx, nitrogen_idx = amide_triplets[0]

        c_atom = mol.GetAtomWithIdx(carbon_idx)
        o_atom = mol.GetAtomWithIdx(oxygen_idx)
        n_atom = mol.GetAtomWithIdx(nitrogen_idx) if nitrogen_idx is not None else None

        if (
            c_atom.GetAtomicNum() != 6
            or o_atom.GetAtomicNum() != 8
            or n_atom is None
            or n_atom.GetAtomicNum() != 7
        ):
            raise ValueError(
                "Context SMARTS selection did not resolve to amide C/O/N atoms"
            )

        all_matches.append(
            {
                "match_index": i,
                "carbon_atom_idx": int(carbon_idx),
                "oxygen_atom_idx": int(oxygen_idx),
                "carbon_atomic_num": int(c_atom.GetAtomicNum()),
                "oxygen_atomic_num": int(o_atom.GetAtomicNum()),
                "nitrogen_atom_idx": int(nitrogen_idx),
                "nitrogen_atomic_num": int(n_atom.GetAtomicNum()),
                "carbon_atom_name": c_atom.GetProp("name"),
                "oxygen_atom_name": o_atom.GetProp("name"),
                "nitrogen_atom_name": n_atom.GetProp("name"),
            }
        )

    if require_unique and len(all_matches) != 1:
        raise ValueError(
            "Context SMARTS matched multiple candidate scissile motifs; expected exactly one. "
            f"match_count={len(all_matches)}. "
            "Refine --scissile-context-smarts or pass --allow-ambiguous-scissile-context "
            "with an explicit --scissile-context-match-index."
        )

    if match_index < 0 or match_index >= len(all_matches):
        raise ValueError(
            f"scissile-context-match-index {match_index} out of range for {len(all_matches)} context SMARTS matches"
        )

    chosen = all_matches[match_index]
    return (
        chosen["carbon_atom_name"],
        chosen["oxygen_atom_name"],
        chosen["nitrogen_atom_name"],
        all_matches,
    )


def extract_chain_sequence_from_pdb(pdb_path: Path, chain_id: str):
    seq = []
    seen = set()
    with pdb_path.open("r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain_id:
                continue
            resname = line[17:20].strip()
            resseq = int(line[22:26])
            ins = line[26].strip()
            key = (resseq, ins)
            if key in seen:
                continue
            seen.add(key)
            seq.append(AA3_TO_AA1.get(resname, "X"))
    if not seq:
        raise ValueError(f"No residues parsed for chain {chain_id} from {pdb_path}")
    return "".join(seq)
