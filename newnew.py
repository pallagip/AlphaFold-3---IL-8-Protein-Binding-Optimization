import numpy as np
import sys
import os
import json
import random
import subprocess
import multiprocessing
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from datetime import datetime
from Bio.PDB import MMCIFParser, PPBuilder

# Add AlphaFold repository to path
alphafold_path = '/app/alphafold3'
sys.path.append(alphafold_path)

# Import AlphaFold modules
from alphafold.common import protein
from alphafold.model import config
from alphafold.model import model
from alphafold.data import pipeline
from alphafold.data.tools import hhsearch
from alphafold.data.parsers import fasta

# Check dependencies
def check_dependencies():
    """Verify required dependencies are available."""
    required_modules = [
        'alphafold3',
        'Bio',
        'numpy',
        'pandas'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        raise ImportError(f"Missing required modules: {', '.join(missing)}")

class Config:
    # Directories
    BASE_DIR = os.path.abspath("protein_binding_search")
    SEQUENCE_DIR = os.path.join(BASE_DIR, "sequences")
    STRUCTURE_DIR = os.path.join(BASE_DIR, "structures")
    DOCKING_DIR = os.path.join(BASE_DIR, "docking")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Files
    TARGET_FILE = os.path.join(SEQUENCE_DIR, "target_il8.fasta")
    RESULTS_CSV = os.path.join(RESULTS_DIR, "binding_results.csv")
    TOP_HITS_CSV = os.path.join(RESULTS_DIR, "top_100_hits.csv")
    
    # Parameters
    NUM_VARIANTS = 1000  # Number of variants to generate per round
    NUM_TOP_HITS = 100   # Number of top hits to keep
    NUM_ROUNDS = 5       # Number of rounds to run
    MUTATION_RATE = 0.05 # Probability of mutation per position
    
    # AlphaFold and FEP parameters
    PAE_CUTOFF = 10.0    # PAE cutoff for interaction analysis
    DIST_CUTOFF = 8.0    # Distance cutoff for interaction analysis
    
    # Paths from environment variables
    ALPHAFOLD_PATH = os.environ.get('ALPHAFOLD_PATH', '/opt/alphafold')
    IPSAE_SCRIPT = os.environ.get('IPSAE_SCRIPT', '/opt/ipsae/ipsae.py')
    FEP_SCRIPT = os.environ.get('FEP_SCRIPT', '/opt/fep/run_fep.py')
    
    # Amino acids
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# IL-8 sequence
IL8_SEQUENCE = "MTSKLAVALLAAFLISAALCEGAVLPRSAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS"

def setup_directories():
    """Create all necessary directories for the pipeline."""
    for directory in [
        Config.BASE_DIR,
        Config.SEQUENCE_DIR,
        Config.STRUCTURE_DIR,
        Config.DOCKING_DIR,
        Config.RESULTS_DIR
    ]:
        os.makedirs(directory, exist_ok=True)
    
    # Create target IL-8 FASTA file
    with open(Config.TARGET_FILE, "w") as f:
        f.write(f">IL8_target\n{IL8_SEQUENCE}\n")
    print(f"Directories created. Base directory: {Config.BASE_DIR}")

def generate_variants(seed_sequences, round_num):
    """Generate protein variants based on seed sequences."""
    variants = []
    variant_dir = os.path.join(Config.SEQUENCE_DIR, f"round_{round_num}")
    os.makedirs(variant_dir, exist_ok=True)
    
    # If it's the first round and no seed sequences provided, create random sequences
    if round_num == 1 and not seed_sequences:
        # Create random sequences of varying lengths (50-150 amino acids)
        for i in range(Config.NUM_VARIANTS):
            length = random.randint(50, 150)
            sequence = ''.join(random.choice(Config.AMINO_ACIDS) for _ in range(length))
            variants.append((f"variant_r1_{i+1}", sequence))
    else:
        # Generate variants based on seed sequences
        variants_per_seed = Config.NUM_VARIANTS // len(seed_sequences)
        for idx, (seed_name, seed_seq) in enumerate(seed_sequences):
            for i in range(variants_per_seed):
                # Create a mutated version of the seed sequence
                new_seq = mutate_sequence(seed_seq, Config.MUTATION_RATE)
                variant_name = f"variant_r{round_num}_{idx+1}_{i+1}"
                variants.append((variant_name, new_seq))
    
    # Write variants to FASTA files
    for name, sequence in variants:
        with open(os.path.join(variant_dir, f"{name}.fasta"), "w") as f:
            f.write(f">{name}\n{sequence}\n")
    print(f"Generated {len(variants)} variants for round {round_num}")
    return variants

def mutate_sequence(sequence, mutation_rate):
    """Introduce random mutations to a sequence."""
    new_sequence = list(sequence)
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            new_sequence[i] = random.choice(Config.AMINO_ACIDS)
    
    # Occasionally add or delete residues (10% chance)
    if random.random() < 0.1 and len(new_sequence) > 30:
        # Delete a random residue
        del_pos = random.randint(0, len(new_sequence) - 1)
        new_sequence.pop(del_pos)
    if random.random() < 0.1:
        # Add a random residue
        add_pos = random.randint(0, len(new_sequence))
        new_sequence.insert(add_pos, random.choice(Config.AMINO_ACIDS))
    
    return ''.join(new_sequence)

def run_alphafold(variant_name, sequence, round_num):
    """Run AlphaFold3 to predict the structure of a variant."""
    try:
        import alphafold3.common.folding_input
        print("AlphaFold3 modules found")
    except ImportError as e:
        print(f"Missing required AlphaFold3 modules: {e}")
        raise
    
    output_dir = os.path.join(Config.STRUCTURE_DIR, f"round_{round_num}", variant_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create input FASTA file
    fasta_file = os.path.join(output_dir, f"{variant_name}.fasta")
    with open(fasta_file, "w") as f:
        f.write(f">{variant_name}\n{sequence}\n")
    
    # Directly call your known AlphaFold script
    cmd = [
        "python",
        os.path.join(Config.ALPHAFOLD_PATH, "run_alphafold.py"),
        "--fasta_paths", fasta_file,
        "--output_dir", output_dir,
        "--model_preset", "monomer",
        "--max_template_date", "2022-01-01",
        "--use_gpu_relax", "True",
        "--num_multimer_predictions_per_model", "1"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=3600,  # 1 hour timeout
            env={
                **os.environ,
                'CUDA_VISIBLE_DEVICES': '0',  # Assuming one GPU
                'TF_FORCE_UNIFIED_MEMORY': '1'
            }
        )
        print(f"Successfully predicted structure for {variant_name}")
        structure_file = os.path.join(output_dir, variant_name, "ranked_0.cif")
        return structure_file
    except subprocess.TimeoutExpired:
        print(f"AlphaFold prediction timed out for {variant_name}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error predicting structure for {variant_name}: {e}")
        print(f"STDOUT: {e.stdout.decode()}")
        print(f"STDERR: {e.stderr.decode()}")
        return None

def run_alphafold_multimer(variant_name, variant_structure, target_structure, round_num):
    """Run AlphaFold-Multimer to predict the interaction between variant and target."""
    output_dir = os.path.join(Config.DOCKING_DIR, f"round_{round_num}", variant_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a combined FASTA file for the complex
    fasta_file = os.path.join(output_dir, f"{variant_name}_complex.fasta")
    
    # Extract sequences from structure files
    variant_seq = extract_sequence_from_structure(variant_structure)
    target_seq = IL8_SEQUENCE
    
    with open(fasta_file, "w") as f:
        f.write(f">{variant_name}\n{variant_seq}\n")
        f.write(f">IL8_target\n{target_seq}\n")
    
    # Directly call your known AlphaFold script for multimer
    cmd = [
        "python",
        os.path.join(Config.ALPHAFOLD_PATH, "run_alphafold.py"),
        "--fasta_paths", fasta_file,
        "--output_dir", output_dir,
        "--model_preset", "multimer",
        "--max_template_date", "2022-01-01",
        "--use_gpu_relax", "True",
        "--num_multimer_predictions_per_model", "1"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=3600,  # 1 hour timeout
            env={
                **os.environ,
                'CUDA_VISIBLE_DEVICES': '0',  # Assuming one GPU
                'TF_FORCE_UNIFIED_MEMORY': '1'
            }
        )
        print(f"Successfully predicted complex for {variant_name} with IL-8")
        complex_structure = os.path.join(output_dir, f"{variant_name}_complex", "ranked_0.cif")
        pae_file = os.path.join(output_dir, f"{variant_name}_complex", "ranked_0_pae.json")
        return complex_structure, pae_file
    except subprocess.TimeoutExpired:
        print(f"AlphaFold-Multimer prediction timed out for {variant_name}")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Error predicting complex for {variant_name}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None, None

def extract_sequence_from_structure(structure_file):
    """Extract the amino acid sequence from the first chain of the first model in an AlphaFold (MMCIF) structure file."""
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("temp", structure_file)
        ppb = PPBuilder()
        # Usually, AlphaFold structures have just 1 model. If multiple, take the first.
        model = structure[0]
        # Concatenate sequences from the first chain we encounter
        for chain in model:
            peptides = ppb.build_peptides(chain)
            seq = "".join(str(peptide.get_sequence()) for peptide in peptides)
            # Return the first chain's sequence
            return seq
        # If no chain or peptides found:
        print(f"No chain or peptides found in {structure_file}")
        return "A" * 100
    except Exception as e:
        print(f"Error extracting sequence from {structure_file}: {e}")
        return "A" * 100

def calculate_binding_energy(complex_structure, pae_file, variant_name, round_num):
    """Calculate binding energy using FEP and IPSAE scoring."""
    output_dir = os.path.join(Config.RESULTS_DIR, f"round_{round_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    # First, use IPSAE to score the interaction
    ipsae_score = run_ipsae(pae_file, complex_structure)
    
    # Then, run FEP calculation for more accurate binding energy
    fep_energy = run_fep(complex_structure, variant_name, output_dir)
    
    return {
        "variant_name": variant_name,
        "ipsae_score": ipsae_score,
        "fep_energy": fep_energy,
        "complex_structure": complex_structure,
        "pae_file": pae_file,
        "round": round_num
    }

def run_ipsae(pae_file, complex_structure):
    """Run IPSAE scoring function to evaluate protein-protein interaction."""
    cmd = [
        "python",
        Config.IPSAE_SCRIPT,
        pae_file,
        complex_structure,
        str(Config.PAE_CUTOFF),
        str(Config.DIST_CUTOFF)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        # Parse the IPSAE output to extract the score
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if "ipSAE" in line:
                # Extract the ipSAE score
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "ipSAE":
                        return float(parts[i+1])
        # If we couldn't find the score, return a default value
        print(f"Warning: Could not parse IPSAE score for {complex_structure}")
        return 0.0
    except subprocess.TimeoutExpired:
        print(f"IPSAE calculation timed out for {complex_structure}")
        return 0.0
    except subprocess.CalledProcessError as e:
        print(f"Error running IPSAE for {complex_structure}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return 0.0

def run_fep(complex_structure, variant_name, output_dir):
    """Run Free Energy Perturbation to calculate binding energy."""
    fep_output = os.path.join(output_dir, f"{variant_name}_fep.log")
    
    cmd = [
        "python",
        Config.FEP_SCRIPT,
        "--structure", complex_structure,
        "--output", fep_output,
        "--steps", "1000",
        "--temperature", "300"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=3600,  # 1 hour timeout
            env={
                **os.environ,
                'CUDA_VISIBLE_DEVICES': '0'  # Assuming one GPU
            }
        )
        # Parse the FEP output to extract the binding energy
        binding_energy = parse_fep_output(fep_output)
        return binding_energy
    except subprocess.TimeoutExpired:
        print(f"FEP calculation timed out for {complex_structure}")
        return 0.0
    except subprocess.CalledProcessError as e:
        print(f"Error running FEP for {complex_structure}: {e}")
        print(f"STDOUT: {e.stdout.decode()}")
        print(f"STDERR: {e.stderr.decode()}")
        return 0.0

def parse_fep_output(fep_log_path):
    """Parse FEP log file and extract binding energy."""
    try:
        with open(fep_log_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "FEP binding energy" in line:
                # Example format: "FEP binding energy: -10.5 kcal/mol"
                parts = line.strip().split(":")
                if len(parts) == 2:
                    return float(parts[1].split()[0])
        # If not found, return 0
        return 0.0
    except:
        return 0.0

if __name__ == "__main__":
    # Check dependencies
    check_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Example: run multiple rounds
    previous_round_variants = []
    for rnd in range(1, Config.NUM_ROUNDS + 1):
        # Generate new variants
        variants = generate_variants(previous_round_variants, rnd)
        
        # Predict structure for each variant
        variant_structures = []
        for vname, seq in variants:
            struct_path = run_alphafold(vname, seq, rnd)
            if struct_path:
                variant_structures.append((vname, struct_path))
        
        # Dock each variant with IL-8
        results = []
        for vname, struct_path in variant_structures:
            complex_struct, pae_json = run_alphafold_multimer(
                vname, struct_path, Config.TARGET_FILE, rnd
            )
            if complex_struct and pae_json:
                score_info = calculate_binding_energy(
                    complex_struct, pae_json, vname, rnd
                )
                results.append(score_info)
        
        # Convert results to DataFrame and sort by FEP or IPSAE
        df = pd.DataFrame(results)
        # Example: sort by FEP energy (lower is better)
        df = df.sort_values(by="fep_energy", ascending=True)
        
        round_dir = os.path.join(Config.RESULTS_DIR, f"round_{rnd}")
        os.makedirs(round_dir, exist_ok=True)
        round_csv = os.path.join(round_dir, f"binding_results_round_{rnd}.csv")
        df.to_csv(round_csv, index=False)
        
        # Pick top hits to seed next round
        top_hits = df.head(Config.NUM_TOP_HITS)
        next_round_variants = []
        for _, row in top_hits.iterrows():
            # Re-extract sequence from the variant structure for the next round
            v_seq = extract_sequence_from_structure(row["complex_structure"])
            next_round_variants.append((row["variant_name"], v_seq))
        
        # Prepare seeds for next round
        previous_round_variants = next_round_variants
    
    print("Pipeline completed.")