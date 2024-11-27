import os
import mdtraj as md
import numpy as np
import pandas as pd
import openmm
import pyarrow as pa
import pyarrow.feather as feather
import click
def get_boresch_restraint_indices(system_file):
    with open(system_file, "r") as f:
        system = openmm.XmlSerializer.deserialize(f.read())
    indices = {"lig_1": [], "lig_2": []}
    for force in system.getForces():
        if isinstance(force, openmm.CustomCompoundBondForce):
            if "lambda_boresch_lig_1" == force.getGlobalParameterName(0):
                for bond_idx in range(force.getNumBonds()):
                    params = force.getBondParameters(bond_idx)
                    indices["lig_1"].extend(params[0]) 
            elif "lambda_boresch_lig_2" == force.getGlobalParameterName(0):
                for bond_idx in range(force.getNumBonds()):
                    params = force.getBondParameters(bond_idx)
                    indices["lig_2"].extend(params[0])
    return indices

def process_trajectory(pdb_file, traj, indices, lig):
    traj0 = md.load(pdb_file, top=pdb_file)
    p1, p2, p3, p4, p5, p6 = indices[lig]

    # distances
    distances = md.compute_distances(traj, [[p3, p4]]) * 10  # to Å
    d0 = md.compute_distances(traj0, [[p3, p4]])[0][0] * 10  

    # angles
    angles1_rad = md.compute_angles(traj, [[p2, p3, p4]])  
    angles2_rad = md.compute_angles(traj, [[p3, p4, p5]]) 
    angles1_deg = angles1_rad * (180 / np.pi)
    angles2_deg = angles2_rad * (180 / np.pi)

    # torsion angles
    torsions1 = md.compute_dihedrals(traj, [[p1, p2, p3, p4]]) * (180 / np.pi)
    torsions2 = md.compute_dihedrals(traj, [[p2, p3, p4, p5]]) * (180 / np.pi)
    torsions3 = md.compute_dihedrals(traj, [[p3, p4, p5, p6]]) * (180 / np.pi)

    # arc lengths
    arc_lengths1 = distances * angles1_rad
    arc_lengths2 = distances * angles2_rad

    return {
        "distances": distances,
        "angles1_deg": angles1_deg,
        "angles2_deg": angles2_deg,
        "torsions1": torsions1,
        "torsions2": torsions2,
        "torsions3": torsions3,
        "arc_lengths1": arc_lengths1,
        "arc_lengths2": arc_lengths2,
        "d0": d0
    }

def state_indices(file_path):

    with pa.OSFile(str(file_path), "rb") as file:
        with pa.RecordBatchStreamReader(file) as reader:
            table = reader.read_all()

    df = table.to_pandas()
    replica_to_state_idx_column = df["replica_to_state_idx"].to_list()

    num_steps = len(replica_to_state_idx_column)
    num_replicas = len(replica_to_state_idx_column[0])
    state_data = {f"Replica {i}": [replica_to_state_idx_column[step][i] for step in range(num_steps)] for i in range(num_replicas)}
    state_idx_df = pd.DataFrame(state_data)
    return state_idx_df

def get_state_trajectory(transformation,  
                        pdb_file, 
                        trajectory_files, 
                        state_idx_df, 
                        state,
                        traj_freq,
                        ):

    num_replicas = len(state_idx_df.columns)
    combined_traj = None
    for replica_idx in range(num_replicas):
        _state_steps = state_idx_df[state_idx_df[f"Replica {replica_idx}"] == state].index.tolist()
        state_steps = []
        for i in _state_steps:
            if i % traj_freq == 0:
                state_steps.append(int(i/traj_freq))
        if not state_steps:
            continue

        traj_file = [f for f in trajectory_files if f"r{replica_idx}" in f][0]
        traj = md.load(traj_file, top=pdb_file)
        # state i frames
        selected_frames = traj.slice(state_steps)
        if combined_traj is None:
            combined_traj = selected_frames
        else:
            combined_traj = combined_traj.join(selected_frames)
    return combined_traj
        

def get_end_states_boresch_terms(root_dir, output_file, save_trajectories=False, traj_freq=100):
    results = []
    trajs = None
    if save_trajectories:
        trajs = {}
    for root, dirs, files in os.walk(root_dir):
        if any(file.endswith(".dcd") for file in files):
            transformation = root.split("/")[-4]
            replica = root.split("/")[-5]
            system_file = os.path.join(root_dir, "inputs", "outputs", replica, transformation, "complex/_setup/system.xml")
            pdb_file = os.path.join(root_dir, "inputs", "outputs", replica, transformation, "complex/_setup/system.pdb")
            trajectory_files = [os.path.join(root, f) for f in files if f.endswith(".dcd")]
            samples_file = os.path.join(root_dir, "inputs", "outputs", replica, transformation, "complex/_sample/samples.arrow")
            state_idx_df = state_indices(samples_file)
            end_states = [0, len(state_idx_df.columns) - 1]
            indices = get_boresch_restraint_indices(system_file)
            for dummy_ligand, state in zip(["lig_2", "lig_1"], end_states):
                                
                traj = get_state_trajectory(transformation,
                                            pdb_file,
                                            trajectory_files,
                                            state_idx_df,
                                            state,
                                            traj_freq)
                if save_trajectories:
                    trajs[f"{transformation}_{replica}_state_{state}"] = traj
                boresch_terms = process_trajectory(pdb_file, traj, indices, dummy_ligand)
                for frame_idx in range(len(boresch_terms["distances"])):
                    results.append({
                        "transformation": transformation,
                        "replica": replica,
                        "state": state,
                        "frame": frame_idx,
                        "distance": boresch_terms["distances"][frame_idx][0],
                        "angle1": boresch_terms["angles1_deg"][frame_idx][0],
                        "angle2": boresch_terms["angles2_deg"][frame_idx][0],
                        "torsion1": boresch_terms["torsions1"][frame_idx][0],
                        "torsion2": boresch_terms["torsions2"][frame_idx][0],
                        "torsion3": boresch_terms["torsions3"][frame_idx][0],
                        "arc_length1": boresch_terms["arc_lengths1"][frame_idx][0],
                        "arc_length2": boresch_terms["arc_lengths2"][frame_idx][0],
                        "d0": boresch_terms["d0"]
                    })

    return results, trajs
@click.command()
@click.option("--root_dir", "-r", type=str, help="Root directory to search for trajectories")
@click.option("--output_file", "-o", type=str, help="Output file for the results")
@click.option("--save_trajectories", "-s", is_flag=True, help="Save end-states trajectories")
@click.option("--traj_freq", "-f", default=100, type=int, help="Frequency of trajectory frames")
def main(root_dir, output_file, save_trajectories, traj_freq):

    boresch_terms, end_state_trajectories = get_end_states_boresch_terms(root_dir, 
                                                 output_file,
                                                 save_trajectories=save_trajectories, 
                                                 traj_freq=traj_freq)
    results_df = pd.DataFrame(boresch_terms)
    results_df.to_csv(output_file, index=False)
    
    if save_trajectories:
        os.makedirs("end_state_trajectories", exist_ok=True)
        for traj_name, traj in end_state_trajectories.items():
            traj.save(f"end_state_trajectories/{traj_name}.dcd")


if __name__ == "__main__":
    main()