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


def prepare_amber_commands(conf):
    boltz_output_folder = Path(conf.boltz.output_dir) / "output"
    commands_amber = []
    for substrate in conf.substrates:
        model_name = f"{substrate['name']}_{conf.enzymes[0]['name']}_model_0"
        amber_input_model = boltz_output_folder / f"boltz_results_{model_name}" / "predictions" / model_name / f"{model_name}_model_0.pdb"
        commands_amber.append(f"{conf.relaxation.amber_command} "
                              f"--input_model={amber_input_model.absolute()} "
                              f"--charge={substrate.charge} "
                              f"--output_folder={conf.relaxation.output_dir}/amber_params_{substrate.name}"
                              )
    print("Example Amber command:")
    print(commands_amber[-1])

    cmds_filename_amber = os.path.join(conf.relaxation.output_dir, "commands_amber.sh")
    with open(cmds_filename_amber, "w") as file:
        file.write("\n".join(commands_amber))


def prepare_relaxation_commands(conf):
    boltz_output_folder = Path(conf.boltz.output_dir) / "output"
    commands_relaxation = []
    for substrate in conf.substrates:
        for enzyme in conf.enzymes:
            for n_model in range(conf.boltz.models_per_sequence):
                model_name = f"{substrate['name']}_{enzyme['name']}_model_{n_model}"
                input_model = boltz_output_folder / f"boltz_results_{model_name}" / "predictions" / model_name / f"{model_name}_model_0.pdb"
                commands_relaxation.append(f"{conf.relaxation.relaxation_command} "
                                      f"--input_model={input_model.absolute()} "
                                      f"--output_folder={conf.relaxation.output_dir} "
                                      f"--amber_params={conf.relaxation.output_dir}/amber_params_{substrate.name} "
                                      f"--residue_number={enzyme['relaxation']['residue_number']}"
                )
    print("Example relaxation command:")
    print(commands_relaxation[-1])

    cmds_filename_relaxation = os.path.join(conf.relaxation.output_dir, "commands_relaxation.sh")
    with open(cmds_filename_relaxation, "w") as file:
        file.write("\n".join(commands_relaxation))

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(conf: HydraConfig) -> None:
    conf.base_dir = os.path.abspath(conf.base_dir)
    Path(conf.relaxation.output_dir).mkdir(parents=True, exist_ok=True)
    logging.info(
        "Preparing relaxation inputs:\n" +
        OmegaConf.to_yaml(conf.boltz, resolve=True)
    )

    prepare_amber_commands(conf)
    prepare_relaxation_commands(conf)


if __name__ == "__main__":
    main()
