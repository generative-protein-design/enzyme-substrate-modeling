from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, TextIO, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
import yaml
from jinja2 import Template, Environment
from copy import deepcopy

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def read_fasta_chains(source) -> List[Dict[str, Any]]:
    """
    Reads FASTA chains from a file path or file-like object.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a FASTA entry with:
            - 'name': the first part of the header (before any commas)
            - other key-value pairs parsed from header parts with 'key=value' format
            - 'chain': the corresponding chain string
    """
    res = {}

    if isinstance(source, str):
        f = open(source, "r")
        close_when_done = True
    else:
        f = source
        close_when_done = False

    try:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = lines[i][1:].strip()
                chain = lines[i + 1].strip()
            else:
                continue

            header_parts = [part.strip() for part in header.split(",")]
            entry = {}

            if header_parts:
                res["name"], type = header_parts[0].split("_")
            res[type] = chain
    finally:
        if close_when_done:
            f.close()

    return [res]


def chain_to_list(chain: str) -> List[Tuple[int, str]]:
    return [(i + 1, char) for i, char in enumerate(chain)]


def list_to_chain(sl: List[Tuple[int, str]]) -> str:
    chars = [t[1] for t in sl]
    return "".join(chars)


def find_index(lst: List[Tuple[int, str]], index: int) -> int:
    for i, x in enumerate(lst):
        if x[0] == index:
            return i
    return -1


def split_chain(chain: str, alpha_beta_split_string: Optional[str]) -> Tuple[str, str]:
    if alpha_beta_split_string:
        alpha, beta = chain.split(alpha_beta_split_string, 1)
        return alpha, alpha_beta_split_string + beta
    else:
        return chain, None


def get_chains_from_fasta_file(source: str | TextIO) -> List[
    Dict[str, Any]]:
    chains = read_fasta_chains(source)
    return chains


def get_chains_from_dir(dir_name: str, pattern: str) -> List[
    Dict[str, Any]]:
    folder = Path(dir_name)
    fa_files = list(folder.glob(f"{pattern}"))
    res = []
    logging.info(f"Processing {len(fa_files)} fasta files from folder {dir_name}")

    for file in fa_files:
        res += get_chains_from_fasta_file(file.resolve().as_posix())

    return res


def combine_constraints(enzyme,substrate):
    combined_constraints = []
    enzyme_constraints = OmegaConf.to_container(
        enzyme["constraints"],
        resolve=True
    )
    substrate_constraints = OmegaConf.to_container(
        substrate["constraints"],
        resolve=True
    )
    for i, e in enumerate(enzyme_constraints):
        s = substrate_constraints[i]
        merged_contact = deepcopy(e["contact"])
        merged_contact.update(s["contact"])
        combined_constraints.append({
                "contact": merged_contact
            })

    return combined_constraints

def render_boltz_input(enzyme, substrate, template_path: str, output_dir: str, conf, msa_file) -> None:
    constraints = combine_constraints(enzyme,substrate)
    context = {
        "alpha_seq": enzyme.chain["alpha"],
        "beta_seq": enzyme.chain["beta"],
        "smiles": substrate.smiles,
        "constraints": constraints,
        "templates": OmegaConf.to_container(enzyme.get("templates", None),resolve=True),
        "properties": OmegaConf.to_container(conf.boltz.properties, resolve=True),
    }
    if not conf.boltz.boltz_params.use_msa_server:
        context["msa_file"] = os.path.join(output_dir, conf.boltz.colabfold.output_folder, msa_file)
    template_path = Path(template_path)
    template_str = template_path.read_text()

    env = Environment()
    env.filters['to_yaml'] = lambda value: yaml.safe_dump(value, default_flow_style=False, sort_keys=False)

    template = env.from_string(template_str)
    return template.render(context)


def save_chain(enzyme: Dict[str, Any], substrate: Dict[str, Any], conf) -> None:
    chain_name = f"{substrate['name']}_{enzyme['name']}"
    output_dir = conf.boltz.input_files_dir
    template_path = conf.boltz.boltz_input_template
    msa_file = enzyme['name']
    rendered_input = render_boltz_input(enzyme, substrate, template_path, output_dir, conf, msa_file)

    for model in range(conf.boltz.models_per_sequence):
        fname = f"{chain_name}_model_{model}.yaml"
        dir_path = Path(output_dir) / conf.boltz.yaml_files_dir
        output_path = dir_path / fname
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered_input)


def copy_sequence(input: str, n_copies: int) -> str:
    inputs = [input] * n_copies
    return ":".join(inputs)


def save_fasta(merge_fasta: bool, enzymes: List[Dict[str, Any]], n_copies: int, output_dir: str):
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    if merge_fasta:
        output_path = dir_path / "fasta.fa"
        output_path.write_text("")
    for enzyme in enzymes:
        if enzyme.chain['beta']:
            chain_str = f"{copy_sequence(enzyme.chain['alpha'], n_copies)}:{copy_sequence(enzyme.chain['beta'], n_copies)}"
        else:
            chain_str = f"{copy_sequence(enzyme.chain['alpha'], n_copies)}"
        fasta_input = (
                f">{enzyme['name']}\n" + chain_str
        )
        if not merge_fasta:
            fname_fasta = f"{enzyme['name']}.fa"
            output_path = dir_path / fname_fasta
            output_path.write_text(fasta_input)
        else:
            with output_path.open("a") as f:
                f.write(fasta_input + "\n")  # add newline if needed


def save_chains(chains: List[Dict[str, Any]], output_dir: str, template_path: str, cif_file: str,
                molecule_smiles: str, conf) -> None:
    logging.info(
        f"Writing {len(chains)} chains/{conf.boltz.models_per_sequence} model(s) each to folder {output_dir}")

    for chain in chains:
        save_chain(chain, output_dir, template_path, cif_file, molecule_smiles, conf)


def prepare_colabfold_search_command(conf):
    folder = Path(conf.boltz.input_files_dir)

    commands_colabfold = []
    fasta_folder = folder / conf.boltz.colabfold.fasta_output_folder
    colabfold_files = list(fasta_folder.glob("*.fa"))

    for file in colabfold_files:
        commands_colabfold.append(f"{conf.boltz.colabfold.search_command} {file} "
                                  f"{conf.boltz.colabfold.database} {folder / conf.boltz.colabfold.output_folder}"
                                  )

    print("Example colabfold_search command:")
    print(commands_colabfold[-1])

    cmds_filename_colabfold = os.path.join(conf.boltz.output_dir, "commands_colabfold_search.sh")
    with open(cmds_filename_colabfold, "w") as file:
        file.write("\n".join(commands_colabfold))


def prepare_msas_convert(conf):
    folder = Path(conf.boltz.input_files_dir)

    commands_msas_convert = []
    colabfold_search_output_folder = folder / conf.boltz.colabfold.output_folder
    for enzyme in conf.enzymes:
        fname = str(colabfold_search_output_folder / enzyme["name"])
        csv_file_alpha = fname + "_alpha.csv"
        csv_file_beta = fname + "_beta.csv"
        msas_file = fname + ".a3m"
        commands_msas_convert.append(f"{conf.boltz.colabfold.convert_command}  --msas_file {msas_file} "
                                     f" --csv_alpha {csv_file_alpha}  --csv_beta {csv_file_beta}"
                                     )

    print("Example msas_convert command:")
    print(commands_msas_convert[-1])

    cmds_filename_colabfold = os.path.join(conf.boltz.output_dir, "commands_msas_convert.sh")
    with open(cmds_filename_colabfold, "w") as file:
        file.write("\n".join(commands_msas_convert))


def prepare_boltz_command(conf):
    input_files_dir = Path(conf.boltz.input_files_dir)
    boltz_yaml_folder = input_files_dir / conf.boltz.yaml_files_dir

    if conf.boltz.batch_processing:
        boltz_files = [boltz_yaml_folder.resolve()]
    else:
        boltz_files = list(boltz_yaml_folder.glob("*.yaml"))

    commands_boltz =[]
    for file in boltz_files:
        commands_boltz.append(f"{conf.boltz.command} {file} "
                              f"--model {conf.boltz.boltz_params.model} "
                              f"--output_format {conf.boltz.boltz_params.output_format} "
                              f"{'--use_msa_server' if conf.boltz.boltz_params.use_msa_server else ''} "
                              f"{'--use_potentials' if conf.boltz.boltz_params.use_potentials else ''} "
                              f"{'--affinity_mw_correction' if conf.boltz.boltz_params.affinity_mw_correction else ''} "
                              f"{'--no_kernels' if conf.boltz.boltz_params.no_kernels else ''} "
                              f"--cache {conf.boltz.boltz_params.cache} "
                              f"--recycling_steps {conf.boltz.boltz_params.recycling_steps} "
                              f"--sampling_steps {conf.boltz.boltz_params.sampling_steps} "
                              f"--diffusion_samples {conf.boltz.boltz_params.diffusion_samples} "
                              f"{conf.boltz.boltz_params.extra_params if conf.boltz.boltz_params.extra_params else ''} "
                              f"--out_dir {conf.boltz.boltz_params.output_dir}/output"
                              )
    print("Example Boltz command:")
    print(commands_boltz[-1])

    cmds_filename_boltz = os.path.join(conf.boltz.output_dir, "commands_boltz2.sh")
    with open(cmds_filename_boltz, "w") as file:
        file.write("\n".join(commands_boltz))


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    conf.base_dir = os.path.abspath(conf.base_dir)

    logging.info(
        "Preparing boltz inputs:\n" +
        OmegaConf.to_yaml(conf.boltz, resolve=True)
    )

    for enzyme in conf.enzymes:
        for substrate in conf.substrates:
            save_chain(enzyme, substrate, conf)

    if not conf.boltz.boltz_params.use_msa_server:
            save_fasta(True, conf.enzymes, 1, os.path.join(conf.boltz.input_files_dir, conf.boltz.colabfold.fasta_output_folder))

    prepare_boltz_command(conf)
    if not conf.boltz.boltz_params.use_msa_server:
        prepare_colabfold_search_command(conf)
        prepare_msas_convert(conf)


if __name__ == "__main__":
    main()
