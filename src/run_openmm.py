import sys
from openmm import unit, LangevinIntegrator
from openmm.app import Simulation, PDBFile, AmberPrmtopFile, AmberInpcrdFile, NoCutoff
import json

prmtop_file = 'complex.prmtop'
inpcrd_file = 'complex.inpcrd'
output_pdb = 'complex_min.pdb'
output_energy = 'energy.json'
try:
    print(f'Loading Amber files: {prmtop_file} and {inpcrd_file}...')
    
    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)

    # non-periodic system
    system = prmtop.createSystem(nonbondedMethod=NoCutoff)

    # Create an integrator
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 0.002*unit.picoseconds)

    # Create a Simulation object
    simulation = Simulation(prmtop.topology, system, integrator)
    simulation.context.setPositions(inpcrd.positions)

    # Calculate the energy before minimization
    initial_state = simulation.context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy()
    print(f"Initial Energy: {initial_energy.value_in_unit(unit.kilojoule_per_mole):.4f} kJ/mol")

    # Run energy minimization
    simulation.minimizeEnergy(
        tolerance = 1.0 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations = 0
    )
    # Get the energy after minimization
    final_state = simulation.context.getState(getEnergy=True)
    final_energy = final_state.getPotentialEnergy()
    print(f"Final Energy: {final_energy.value_in_unit(unit.kilojoule_per_mole):.4f} kJ/mol")

    data = {"initial_energy": initial_energy.value_in_unit(unit.kilojoule_per_mole),
            "final_energy": final_energy.value_in_unit(unit.kilojoule_per_mole)}

    with open(output_energy, 'w') as f:
        json.dump(data, f, indent=4)

    # Write the final structure
    with open(output_pdb, 'w') as f:
        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, positions, f)

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure {prmtop_file} and {inpcrd_file} are in the current directory.")
except Exception as e:
    print(f"An error occurred: {e}")

