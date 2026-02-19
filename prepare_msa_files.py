import argparse
from dataclasses import dataclass

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


@dataclass
class Seq:
    name: str
    sequence: str


def get_residuelens_stoichiometries(lines: list[str]) -> tuple[list[int], list[int]]:
    """Get residue lengths and stoichiometries from msa file.
    Args:
        lines: list[str]
            Lines of input msa file
    Returns:
        residue_lens: list[int]
            Residue lengths of each polypeptide chain
        stoichiometries: list[int]
            Stoichiomerties of each polypeptide chain
    """
    if lines[0].startswith("#"):
        residue_lens_, stoichiometries_ = lines[0].split("\t")
        residue_lens = list(map(int, residue_lens_.lstrip("#").split(",")))
        stoichiometries = list(map(int, stoichiometries_.split(",")))
    else:
        # If the first line does not start with '#',
        # get the residue length from the first sequence.
        # Always assume a monomer prediction.
        if not lines[0].startswith(">"):
            raise ValueError(
                "The first line of the input MSA file must start with '#' or '>'."
            )
        residue_lens = [len(lines[1].strip())]
        stoichiometries = [1]
    return residue_lens, stoichiometries


def split_a3msequences(residue_lens, line) -> list[str]:
    """Split a3m sequences into a list of a3m sequences.
    Note: The a3m-format MSA file represents inserted residues with lowercase.
    The first line (starting with '#') of the MSA file contains residue lengths
    and stoichiometries of each polypeptide chain.
    From the second line, the first sequence is the query.
    After this, the paired MSA blocks are followed by the unpaired MSA.
    Args:
        residue_lens: list[int]
            Residue lengths of each polypeptide chain
        line: str
            A3M sequences
    Returns:
        a3msequences: list[str]
            A3M sequences, len(a3msequences) should be the same as len(residue_lens).
    """
    a3msequences = [""] * len(residue_lens)
    i = 0
    count = 0
    current_residue = []

    for char in line:
        current_residue.append(char)
        if char == "-" or char.isupper():
            count += 1
        if count == residue_lens[i]:
            a3msequences[i] = "".join(current_residue)
            current_residue = []
            count = 0
            i += 1
            if i == len(residue_lens):
                break

    if current_residue and i < len(residue_lens):
        a3msequences[i] = "".join(current_residue)

    return a3msequences


def get_paired_and_unpaired_msa(
    lines: list[str], residue_lens: list[int], cardinality: int
) -> tuple[list[list[Seq]], list[list[Seq]]]:
    """Get paired and unpaired MSAs from input MSA file.
    Args:
        lines: list[str]
            Lines of input MSA file
        residue_lens: list[int]
            Residue lengths of each polypeptide chain
        cardinality: int
            Number of polypeptide chains
        query_seqnames: list[int]
            Query sequence names
    Returns:
        pairedmsas: list[list[Seq]]
            Paired MSAs, len(pairedmsa) should be the cardinality.
            If cardinality is 1, pairedmsas returns [[Seq("", "")]].
        unpairedmsas: list[list[Seq]]
            Unpaired MSAs, len(unpairedmsa) should be the cardinality.
    """
    pairedmsas: list[list[Seq]] = [[] for _ in range(cardinality)]
    unpairedmsas: list[list[Seq]] = [[] for _ in range(cardinality)]
    pairedflag = False
    unpairedflag = False
    seen = False
    seqnames_seen = []
    query_seqnames = [int(101 + i) for i in range(cardinality)]
    chain = -1
    start = 1 if lines[0].startswith("#") else 0
    for line in lines[start:]:
        if line.startswith(">"):
            if line not in seqnames_seen:
                seqnames_seen.append(line)
            else:
                seen = True
                continue
            if cardinality > 1 and line.startswith(
                ">" + "\t".join(map(str, query_seqnames)) + "\n"
            ):
                pairedflag = True
                unpairedflag = False
            elif any(line.startswith(f">{seq}\n") for seq in query_seqnames):
                pairedflag = False
                unpairedflag = True
                chain += 1
            seqname = line
        else:
            if seen:
                seen = False
                continue
            if pairedflag:
                a3mseqs = split_a3msequences(residue_lens, line)
                for i in range(cardinality):
                    pairedmsas[i].append(Seq(seqname, a3mseqs[i]))

            elif unpairedflag:
                a3mseqs = split_a3msequences(residue_lens, line)
                for i in range(cardinality):
                    # Remove all-gapped sequences
                    if a3mseqs[i] == "-" * residue_lens[i]:
                        continue
                    unpairedmsas[i].append(Seq(seqname, a3mseqs[i]))
            else:
                raise ValueError("Flag must be either paired or unpaired.")
    return pairedmsas, unpairedmsas


def get_msas(
    inputmsafile: str
) -> None:
    """Write AlphaFold3 input JSON file from a3m-format MSA file.

    Args:
        inputmsafile (str): Input MSA file path.
        pairedmsas (list[list[Seq]]): Paired MSAs.
        unpairedmsas (list[list[Seq]]): Unpaired MSAs.
    """
    with open(inputmsafile, "r") as f:
        lines = f.readlines()
    residue_lens, stoichiometries = get_residuelens_stoichiometries(lines)
    if len(residue_lens) != len(stoichiometries):
        raise ValueError("Length of residue_lens and stoichiometries must be the same.")
    cardinality = len(residue_lens)
    logging.info(
        f"The input MSA file contains {cardinality} distinct polypeptide chains."
    )
    logging.info(f"Residue lengths: {residue_lens}")
    logging.info(f"Stoichiometries: {stoichiometries}")
    pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
        lines, residue_lens, cardinality
    )
    return pairedmsas, unpairedmsas

MAX_PAIRED_SEQS = 8192
MAX_MSA_SEQS = 16384

def write_csv(paired_msas, unpaired_msas, file1, file2):
    fnames = [file1, file2]
    for idx in range(len(paired_msas)):
        # Get paired sequences
        paired = [ s.sequence for s in paired_msas[idx]]
        paired = paired[: MAX_PAIRED_SEQS]
        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = [ s.sequence for s in unpaired_msas[idx]]
        unpaired = unpaired[: (MAX_MSA_SEQS - len(paired))]

        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        with open(fnames[idx], "w") as f:
            f.write("\n".join(csv_str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Process MSAs and CSVs")
    parser.add_argument("--msas_file", required=True, help="Path to MSA file (.a3m)")
    parser.add_argument("--csv_alpha", required=True, help="Output for alpha CSV file")
    parser.add_argument("--csv_beta", required=True, help="Output for beta CSV file")

    args = parser.parse_args()

    logging.info(f"Converting {args.msas_file}...")

    pairedmsas, unpairedmsas = get_msas(args.msas_file)
    write_csv(pairedmsas, unpairedmsas, args.csv_alpha,args.csv_beta)

    logging.info(f"Finished converting {args.msas_file}")

if __name__ == "__main__":
    main()
