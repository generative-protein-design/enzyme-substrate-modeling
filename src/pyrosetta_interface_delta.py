import pyrosetta
from pyrosetta import Pose
from pyrosetta.teaching import get_score_function
import pyrosetta.rosetta.core.import_pose
import json

# Initialize PyRosetta with the ligand .params file
pyrosetta.init('-extra_res_fa LIG.params')

# Load the protein-ligand complex from PDB
complex_pose = Pose()
pyrosetta.rosetta.core.import_pose.pose_from_file(complex_pose, 'complex_min.pdb')

# Make a copy of the pose to separate the components later
separated_pose = complex_pose.clone()

sfxn = get_score_function(True)

# Calculate the energy of the complex
E_complex = sfxn(complex_pose)
print(f"Energy of complex: {E_complex:.2f}")

# Calculate the energies of the separated protein and ligand 
# Get the index of the last residue, which should always be the ligand
ligand_res_idx = separated_pose.total_residue()

if ligand_res_idx == 0:
    raise ValueError("Pose is empty.")

# Create a new pose for the isolated ligand by cloning the residue
ligand_pose = Pose()
ligand_residue = separated_pose.residue(ligand_res_idx)
ligand_pose.append_residue_by_bond(ligand_residue.clone(), True)

# Delete the ligand from separated_pose to leave only the protein
separated_pose.delete_residue_range_slow(ligand_res_idx, ligand_res_idx)

# Score the separated components
E_protein = sfxn(separated_pose)
E_ligand = sfxn(ligand_pose)
E_separated = E_protein + E_ligand
print(f"Energy of separated protein: {E_protein:.2f}")
print(f"Energy of separated ligand: {E_ligand:.2f}")
print(f"Total energy of separated components: {E_separated:.2f}")

# Calculate interface_delta
interface_delta = E_complex - E_separated
print(f"Interface delta: {interface_delta:.2f}")

output_energy = 'energy.json'
with open(output_energy, "r") as f:
    data = json.load(f)

data.update({
    "e_protein": f"{E_protein:.2f}",
    "e_ligand": f"{E_ligand:.2f}",
    "e_separated": f"{E_separated:.2f}",
    "interface_delta": f"{interface_delta:.2f}"
})

# Write results
with open(output_energy, "w") as f:
    json.dump(data, f, indent=4)
