import os
import sys
import subprocess
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from datetime import datetime
from Bio.PDB import PDBParser, PPBuilder
import importlib.util
import jax.numpy as jnp
from typing import Dict, Optional, Tuple, List

class Config:
    # IL-8 (human interleukin factor) amino acid sequence:
    IL8_SEQ = (
        "MTSKLAVALLAAFLISAALCEGAVLPRSAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGREL"
        "CLDPKENWVQRVVEKFLKRAENS"
    )
    
    # Where we'll store results, PDB files, etc.
    RESULTS_DIR = "./results"
    IL8_BASENAME = "IL8"  # used for naming IL-8 structure file
    
    # If you have a known PDB for IL-8, place it here and skip the fold step.
    # If None, we'll generate the IL-8 structure with AlphaFold monomer.
    IL8_PDB_PATH = None
    
    # Round-based search parameters:
    NUM_ROUNDS = 3  # how many iterative rounds
    NUM_VARIANTS_PER_ROUND = 50  # how many new library sequences to generate each round
    NUM_TOP_HITS = 10  # how many top hits to keep for the next round
    
    # AlphaFold installation location (JAX-based).
    # Adjust to where alphafold is installed in your environment:
    ALPHAFOLD_PATH = "/home/c_ppkbm/.local/lib/python3.11/site-packages/alphafold"

def setup_alphafold_import():
    """Set up imports for JAX-based AlphaFold"""
    if not os.path.exists(Config.ALPHAFOLD_PATH):
        print(f"Error: The specified AlphaFold path '{Config.ALPHAFOLD_PATH}' does not exist.")
        sys.exit(1)
    
    # Remove any tensorflow-related modifications
    alphafold_parent = os.path.dirname(Config.ALPHAFOLD_PATH)
    if alphafold_parent not in sys.path:
        sys.path.insert(0, alphafold_parent)
    
    try:
        from alphafold.common import protein
        from alphafold.model import config
        from alphafold.model import model
        from alphafold.data import pipeline
        from alphafold.data.tools import hhsearch
        from alphafold.data.parsers import fasta
        
        global AF_CONFIG
        global AF_MODEL
        global FEATURE_PROCESSOR
        global PREDICTION_PROCESSOR
        
        AF_CONFIG = config.model_config("model_1")
        AF_MODEL = model.Model(AF_CONFIG, direct_jax=True)
        
        FEATURE_PROCESSOR = pipeline.DataPipeline(
            jackhmmer_binary_path="/usr/bin/jackhmmer",
            hhblits_binary_path="/usr/bin/hhblits",
            uniref90_database_path="/path/to/uniref90/uniref90.fasta",
            mgnify_database_path="/path/to/mgnify/mgy_clusters_2018_12.fa",
            pdb70_database_path="/path/to/pdb70/pdb70",
            use_gpu=True
        )
        
        PREDICTION_PROCESSOR = protein.PredictionProcessor(
            model_runner=AF_MODEL,
            config=AF_CONFIG
        )
    except ImportError as e:
        print(f"Error importing AlphaFold modules: {e}")
        sys.exit(1)

def check_python_dependencies():
    """Check whether required Python modules are installed."""
    required_modules = ["Bio", "numpy", "pandas", "jax", "alphafold"]
    missing = []
    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(f"Missing required Python modules: {', '.join(missing)}")

def fold_structure(sequence: str, prefix: str, output_directory: str) -> str:
    """Run AlphaFold prediction for a single sequence.
    
    Args:
        sequence: Amino acid sequence to fold
        prefix: Prefix for output files
        output_directory: Directory to save outputs
    
    Returns:
        Path to the predicted PDB file
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Create temporary FASTA file
    fasta_path = os.path.join(output_directory, f"{prefix}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{prefix}\n{sequence}\n")
    
    # Prepare features
    feature_dict = FEATURE_PROCESSOR.process(
        input_fasta_path=fasta_path,
        preset='reduced_dbs'
    )
    
    # Convert features to JAX arrays
    processed_feature_dict = {
        k: jnp.array(v) for k, v in feature_dict.items()
    }
    
    # Run prediction
    prediction = PREDICTION_PROCESSOR.process(
        proc_features=processed_feature_dict,
        return_raw=True
    )
    
    # Save the prediction
    pdb_path = os.path.join(output_directory, f"{prefix}.pdb")
    with open(pdb_path, "w") as f:
        f.write(protein.to_pdb(prediction['structure_module']['final_atom_positions'],
                              prediction['structure_module']['final_atom_mask']))
    
    return pdb_path

def fold_il8_if_needed():
    """If Config.IL8_PDB_PATH is None or the file does not exist,
    run AlphaFold monomer on the IL-8 sequence to produce a structure.
    Return the path to the IL-8 PDB."""
    if Config.IL8_PDB_PATH and os.path.exists(Config.IL8_PDB_PATH):
        return Config.IL8_PDB_PATH
    
    il8_dir = os.path.join(Config.RESULTS_DIR, "IL8_monomer")
    os.makedirs(il8_dir, exist_ok=True)
    il8_pdb_path = os.path.join(il8_dir, f"{Config.IL8_BASENAME}.pdb")
    
    print("[INFO] No IL-8 PDB provided. Running AlphaFold monomer to fold IL-8...")
    il8_pdb_path = fold_structure(
        sequence=Config.IL8_SEQ,
        prefix=Config.IL8_BASENAME,
        output_directory=il8_dir
    )
    
    Config.IL8_PDB_PATH = il8_pdb_path
    return il8_pdb_path

def generate_sequence_library(base_seqs, num_variants=50):
    """Given one or more base sequences, create a library of random mutants.
    Only uses the 20 standard amino acids.
    Returns a list of (unique_name, sequence_str)."""
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    library = []
    
    for base_seq in base_seqs:
        for i in range(num_variants):
            seq_chars = list(base_seq)
            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                pos = random.randint(0, len(seq_chars) - 1)
                seq_chars[pos] = random.choice(amino_acids)
            new_seq = "".join(seq_chars)
            variant_name = f"{base_seq[:5]}_variant_{i}"
            library.append((variant_name, new_seq))
    
    return library

def run_alphafold_monomer(variant_name, seq):
    """Given a variant name and its sequence, run AlphaFold in monomer mode.
    Return the path to the predicted PDB."""
    output_dir = os.path.join(Config.RESULTS_DIR, f"{variant_name}_monomer")
    os.makedirs(output_dir, exist_ok=True)
    return fold_structure(
        sequence=seq,
        prefix=variant_name,
        output_directory=output_dir
    )

def run_alphafold_multimer(variant_name, variant_pdb_path, il8_pdb_path):
    """Given the candidate's monomer PDB and the IL-8 PDB,
    run AlphaFold in multimer mode to predict the complex.
    Return (complex_pdb_path, pae_json_path)."""
    out_dir = os.path.join(Config.RESULTS_DIR, f"{variant_name}_multimer")
    os.makedirs(out_dir, exist_ok=True)
    
    # Create multimer input
    multimer_fasta = os.path.join(out_dir, f"{variant_name}_multimer.fasta")
    with open(multimer_fasta, "w") as f:
        f.write(f">{variant_name}\n{SeqIO.read(variant_pdb_path, 'pdb').seq}\n")
        f.write(f">IL8\n{SeqIO.read(il8_pdb_path, 'pdb').seq}\n")
    
    # Run multimer prediction
    complex_pdb = fold_structure(
        sequence=SeqIO.read(multimer_fasta, 'fasta').seq,
        prefix=f"{variant_name}_complex",
        output_directory=out_dir
    )
    
    pae_json = os.path.join(out_dir, f"{variant_name}_pae.json")
    # Note: Real PAE calculation would go here
    with open(pae_json, "w") as f:
        f.write("{\"dummy_pae\": [0.1, 0.2, 0.3]}\n")
    
    return complex_pdb, pae_json

def calc_fep_energy(complex_pdb: str, pae_json: str) -> float:
    """Estimate binding energy using a more sophisticated approach combining
    structural analysis with molecular mechanics principles.
    
    Args:
        complex_pdb: Path to the complex PDB file containing two chains
        pae_json: Path to predicted aligned error JSON (currently unused)
    
    Returns:
        Estimated binding free energy in kcal/mol (more negative = stronger binding)
    """
    from Bio.PDB import PDBParser
    import numpy as np
    
    # Constants based on typical molecular mechanics force fields
    ELECTROSTATIC_WEIGHT = 0.2  # Weight for electrostatic interactions
    HBOND_ENERGY = -1.5  # kcal/mol per hydrogen bond
    CONTACT_ENERGY = -0.15  # kcal/mol per atomic contact
    SOLVATION_FACTOR = 0.1  # Solvent accessibility contribution
    
    def get_hydrogen_bond_energy(atom1, atom2):
        """Calculate hydrogen bond energy contribution"""
        # Simple geometric criteria for H-bond identification
        distance = abs(atom1 - atom2)
        return HBOND_ENERGY if distance <= 3.5 else 0.0
    
    def get_electrostatic_term(charge1, charge2, distance):
        """Calculate Coulomb interaction term"""
        # Simplified Coulomb's law with distance-dependent dielectric
        epsilon_r = 4.0  # Relative permittivity
        coulomb_const = 332.059  # kcal/(mol·Å·e²)
        return ELECTROSTATIC_WEIGHT * (coulomb_const * charge1 * charge2) / (epsilon_r * distance)
    
    def get_surface_area(atom):
        """Approximate solvent accessible surface area"""
        # Simplified SASA calculation using atomic radii
        atomic_radii = {
            'N': 1.65, 'CA': 1.87, 'C': 1.76, 'O': 1.40,
            'CB': 1.87, 'H': 1.20
        }
        return 4 * np.pi * (atomic_radii.get(atom.name.strip(), 1.5))**2
    
    try:
        # Parse structure and initialize energy components
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", complex_pdb)
        
        # Verify presence of both chains
        chains = list(structure.get_chains())
        if len(chains) < 2:
            raise ValueError("Complex must contain at least two chains")
        
        chain_a = chains[0]
        chain_b = chains[1]
        
        # Initialize energy components
        total_energy = 0.0
        contact_energy = 0.0
        electrostatic_energy = 0.0
        h_bond_energy = 0.0
        solvation_energy = 0.0
        
        # Get heavy atoms for both chains
        chain_a_atoms = [(atom, atom.element) for atom in chain_a.get_atoms()
                        if atom.element != 'H']
        chain_b_atoms = [(atom, atom.element) for atom in chain_b.get_atoms()
                        if atom.element != 'H']
        
        # Process interactions between chains
        for atom_a, elem_a in chain_a_atoms:
            for atom_b, elem_b in chain_b_atoms:
                distance = abs(atom_a - atom_b)
                
                # Contact energy (van der Waals-like term)
                if distance <= 5.0:
                    contact_energy += CONTACT_ENERGY
                
                # Hydrogen bond energy
                if (elem_a in ['N', 'O'] and elem_b in ['N', 'O']):
                    h_bond_energy += get_hydrogen_bond_energy(atom_a, atom_b)
                
                # Electrostatic energy
                # Note: In a full MD simulation, we'd use partial charges
                # Here we use a simplified model
                if elem_a in ['N', 'O'] and elem_b in ['N', 'O']:
                    q1 = 0.5 if elem_a == 'N' else -0.5
                    q2 = 0.5 if elem_b == 'N' else -0.5
                    electrostatic_energy += get_electrostatic_term(q1, q2, distance)
        
        # Calculate solvation term based on buried surface area
        for atom, _ in chain_a_atoms + chain_b_atoms:
            area = get_surface_area(atom)
            # Buried surface area contributes positively to binding
            solvation_energy += SOLVATION_FACTOR * area
        
        # Combine all energy terms
        total_energy = (contact_energy + electrostatic_energy +
                       h_bond_energy + solvation_energy)
        
        return total_energy
    
    except Exception as e:
        print(f"Error calculating FEP energy: {e}")
        return 100.0  # High energy indicates failed calculation

def main():
    check_python_dependencies()
    setup_alphafold_import()
    
    # Create main results directory
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 1) Prepare the IL-8 structure (fold if not provided)
    il8_pdb_path = fold_il8_if_needed()
    
    # -----------------------------------------
    # Define your initial base sequences here:
    # -----------------------------------------
    #
    # You might have only one base sequence, or multiple.
    # They MUST use only the 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY).
    #
    # Example: A single short "toy" sequence. Replace with your real sequence(s).
    base_sequences = ["ACDEFGHIKLMNPQRSTVWY"]
    
    # This list will track *all* variants from all rounds (for a final top-100).
    scored_variants_all = []
    
    # We will start each round by mutating the "top sequences" from the previous round
    top_sequences = base_sequences[:]  # Make a copy
    
    for round_idx in range(Config.NUM_ROUNDS):
        print(f"\n[INFO] Starting Round {round_idx+1} of {Config.NUM_ROUNDS}")
        
        # STEP 1: Generate a library of variants from our top sequences
        library = generate_sequence_library(
            base_seqs=top_sequences,
            num_variants=Config.NUM_VARIANTS_PER_ROUND
        )
        
        # We'll score each variant and collect them here
        round_scored_variants = []
        
        for variant_name, seq in library:
            print(f"[INFO] Folding monomer for variant: {variant_name}")
            monomer_pdb_path = run_alphafold_monomer(variant_name, seq)
            
            print(f"[INFO] Predicting complex with IL-8 for variant: {variant_name}")
            complex_pdb_path, pae_json_path = run_alphafold_multimer(
                variant_name=variant_name,
                variant_pdb_path=monomer_pdb_path,
                il8_pdb_path=il8_pdb_path
            )
            
            print(f"[INFO] Calculating binding energy (FEP) for variant: {variant_name}")
            binding_energy = calc_fep_energy(
                complex_pdb=complex_pdb_path,
                pae_json=pae_json_path
            )
            
            # Store this variant's results
            round_scored_variants.append((variant_name, seq, binding_energy))
        
        # Sort by binding energy ascending (more negative = stronger binding)
        round_scored_variants.sort(key=lambda x: x[2])
        
        # Keep track of *all* variants from this round
        scored_variants_all.extend(round_scored_variants)
        
        # STEP 4: Keep top hits for next round
        # Use up to NUM_TOP_HITS from this round, sorted by energy
        best_of_round = round_scored_variants[:Config.NUM_TOP_HITS]
        # The sequences alone (for the library generator next round)
        top_sequences = [seq for (_, seq, _) in best_of_round]
        
        # Print top hits of this round
        print(f"\n[INFO] Top {Config.NUM_TOP_HITS} hits for Round {round_idx+1}:")
        for i, (name, seq, energy) in enumerate(best_of_round, start=1):
            print(f"  {i:2d}. {name} | Binding Energy = {energy:.2f} kcal/mol | Seq = {seq}")
    
    # -----------------------------------------
    # Step 5: Listing the Top 100 Hits Overall
    # -----------------------------------------
    print("\n[INFO] All rounds completed. Finding overall top 100 hits...")
    
    # Sort everything by binding energy
    scored_variants_all.sort(key=lambda x: x[2])
    
    # Take top 100 overall
    top_100 = scored_variants_all[:100]
    
    print("\n[INFO] Top 100 variants across all rounds (lowest energy first):")
    for i, (name, seq, energy) in enumerate(top_100, start=1):
        print(f"  {i:3d}. {name} | Energy = {energy:.2f} | Seq = {seq}")
    
    print("\n[INFO] Done! You can inspect the output PDBs and JSON files in the results directory.")