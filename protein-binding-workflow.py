import os
import random
import json
import subprocess
import time
from itertools import product
import argparse
from datetime import datetime

# The 20 standard amino acids
AA_CODES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

class ProteinBindingOptimizer:
    def __init__(self, target_pdb, output_dir="optimization_results", 
                 max_length=32, min_length=4, alphafold_path=None):
        """
        Initialize the protein binding optimizer.
        
        Args:
            target_pdb: Path to the target protein PDB file (IL-8 in this case)
            output_dir: Directory to store all results
            max_length: Maximum length of generated sequences
            min_length: Minimum length of generated sequences
            alphafold_path: Path to AlphaFold installation
        """
        self.target_pdb = target_pdb
        self.output_dir = output_dir
        self.max_length = max_length
        self.min_length = min_length
        self.alphafold_path = alphafold_path or "/path/to/alphafold"
        
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(output_dir, "optimization_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"Protein Binding Optimization Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target: {target_pdb}\n\n")
    
    def log(self, message):
        """Add a message to the log file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def generate_variants(self, seed_sequences, num_variants=100, max_mutations=3):
        """Generate variants of seed sequences with random mutations."""
        self.log(f"Generating {num_variants} variants from {len(seed_sequences)} seed sequences")
        variants = []
        
        for seed in seed_sequences:
            # Add the original sequence
            variants.append(seed)
            
            # Generate variants
            for _ in range(max(1, num_variants // len(seed_sequences))):
                variant = list(seed)
                num_mutations = random.randint(1, min(max_mutations, len(seed)))
                
                # Positions to mutate
                positions = random.sample(range(len(seed)), num_mutations)
                
                for pos in positions:
                    # Choose a different amino acid
                    new_aa = random.choice([aa for aa in AA_CODES if aa != variant[pos]])
                    variant[pos] = new_aa
                
                variants.append(''.join(variant))
        
        return list(set(variants))  # Remove duplicates
    
    def generate_random_sequences(self, num_sequences=100):
        """Generate completely random sequences."""
        self.log(f"Generating {num_sequences} random sequences")
        sequences = []
        
        for _ in range(num_sequences):
            length = random.randint(self.min_length, self.max_length)
            sequence = ''.join(random.choice(AA_CODES) for _ in range(length))
            sequences.append(sequence)
        
        return sequences
    
    def save_sequences_to_json(self, sequences, filename):
        """Save sequences to a JSON file."""
        data = {"sequences": [{"id": f"seq_{i}", "sequence": seq} for i, seq in enumerate(sequences)]}
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.log(f"Saved {len(sequences)} sequences to {filename}")
        return filename
    
    def run_alphafold_batch(self, json_file, output_dir):
        """
        Run AlphaFold on a batch of sequences from a JSON file.
        """
        self.log(f"Running AlphaFold batch prediction on sequences from {json_file}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load sequences from JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        sequences = data["sequences"]
        results = {}
        
        for seq_data in sequences:
            seq_id = seq_data["id"]
            sequence = seq_data["sequence"]
            
            # Create a temporary FASTA file for this sequence
            fasta_file = os.path.join(output_dir, f"temp_{seq_id}.fasta")
            with open(fasta_file, 'w') as f:
                f.write(f">{seq_id}\n{sequence}\n")
            
            # Set up the output directory for this sequence
            seq_output_dir = os.path.join(output_dir, seq_id)
            os.makedirs(seq_output_dir, exist_ok=True)
            
            # Run AlphaFold
            # Note: This command needs to be adjusted based on your specific AlphaFold setup
            cmd = [
                "python", os.path.join(self.alphafold_path, "run_alphafold.py"),
                "--fasta_paths", fasta_file,
                "--output_dir", seq_output_dir,
                "--model_preset", "monomer",
                "--max_template_date=2023-01-01"
            ]
            
            try:
                self.log(f"Processing sequence {seq_id}: {sequence}")
                
                # This is where you would actually run AlphaFold
                # For demonstration, we'll simulate success without running
                # subprocess.run(cmd, check=True)
                
                # Simulate a short delay
                time.sleep(0.1)
                
                # Create a placeholder PDB file
                pdb_path = os.path.join(seq_output_dir, "ranked_0.pdb")
                with open(pdb_path, 'w') as f:
                    f.write(f"SIMULATED PDB FOR {seq_id}\n")
                
                results[seq_id] = {
                    "status": "success",
                    "output_dir": seq_output_dir,
                    "pdb_path": pdb_path,
                    "sequence": sequence
                }
            except Exception as e:
                self.log(f"Error processing sequence {seq_id}: {str(e)}")
                results[seq_id] = {"status": "failed", "error": str(e)}
            
            # Clean up
            if os.path.exists(fasta_file):
                os.remove(fasta_file)
        
        # Save results summary
        results_file = os.path.join(output_dir, "results_summary.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"AlphaFold batch prediction completed, results saved to {results_file}")
        return results
    
    def run_docking_simulations(self, alphafold_results, output_dir):
        """
        Run protein-protein docking simulations.
        """
        self.log(f"Setting up protein-protein docking simulations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the AlphaFold results
        if isinstance(alphafold_results, str):
            with open(alphafold_results, 'r') as f:
                results = json.load(f)
        else:
            results = alphafold_results
        
        docking_results = {}
        
        for seq_id, result in results.items():
            if result.get("status") == "success":
                pdb_path = result["pdb_path"]
                sequence = result.get("sequence", "Unknown")
                
                # Create directory for this docking job
                job_dir = os.path.join(output_dir, seq_id)
                os.makedirs(job_dir, exist_ok=True)
                
                # In a real implementation, you would run your docking software here
                # For example, HADDOCK, PyRosetta, or another tool
                
                self.log(f"Simulating docking for {seq_id}: {sequence}")
                
                # Simulate binding energy calculation
                # In a real implementation, parse the actual docking results
                
                # For demonstration, generate a random binding energy that somewhat correlates with sequence length
                # (slightly favoring longer sequences, with some randomness)
                sequence_length = len(sequence)
                base_energy = -5 - (sequence_length / 10)  # More negative is better
                random_factor = random.uniform(-3, 3)  # Add some randomness
                binding_energy = base_energy + random_factor
                
                docking_results[seq_id] = {
                    "binding_energy": binding_energy,
                    "output_dir": job_dir,
                    "sequence": sequence
                }
        
        # Sort results by binding energy (lowest first)
        sorted_results = sorted(
            [(seq_id, data) for seq_id, data in docking_results.items()],
            key=lambda x: x[1]["binding_energy"]
        )
        
        # Format for saving
        formatted_results = {
            "results": [
                {
                    "seq_id": seq_id,
                    "binding_energy": data["binding_energy"],
                    "sequence": data["sequence"]
                }
                for seq_id, data in sorted_results
            ]
        }
        
        # Save results
        results_file = os.path.join(output_dir, "docking_results.json")
        with open(results_file, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        
        self.log(f"Docking simulations completed, results saved to {results_file}")
        
        # Print top 10 results
        self.log("\nTop 10 binding candidates:")
        for i, (seq_id, data) in enumerate(sorted_results[:10]):
            self.log(f"{i+1}. {seq_id}: {data['sequence']} (Binding Energy: {data['binding_energy']:.2f})")
        
        return formatted_results
    
    def run_iteration(self, iteration, seed_sequences=None, num_variants=100, num_random=50):
        """Run a complete iteration of the optimization process."""
        self.log(f"\n--- Starting Iteration {iteration} ---")
        
        # Create directory for this iteration
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Step 1: Generate sequences
        if seed_sequences:
            variants = self.generate_variants(seed_sequences, num_variants=num_variants)
            if num_random > 0:
                random_seqs = self.generate_random_sequences(num_sequences=num_random)
                all_sequences = variants + random_seqs
            else:
                all_sequences = variants
        else:
            # No seeds provided, generate only random sequences
            all_sequences = self.generate_random_sequences(num_sequences=num_variants + num_random)
        
        # Save sequences to JSON
        json_file = self.save_sequences_to_json(
            all_sequences, 
            os.path.join(iteration_dir, "protein_library.json")
        )
        
        # Step 2: Run AlphaFold
        alphafold_output_dir = os.path.join(iteration_dir, "alphafold_results")
        alphafold_results = self.run_alphafold_batch(json_file, alphafold_output_dir)
        
        # Step 3: Run docking simulations
        docking_output_dir = os.path.join(iteration_dir, "docking_results")
        docking_results = self.run_docking_simulations(alphafold_results, docking_output_dir)
        
        self.log(f"--- Completed Iteration {iteration} ---\n")
        return docking_results
    
    def run_optimization(self, seed_sequences=["AYGE"], num_iterations=5, top_n=10):
        """
        Run the complete optimization process for multiple iterations.
        """
        self.log(f"Starting optimization process with {num_iterations} iterations")
        self.log(f"Initial seed sequences: {seed_sequences}")
        
        current_seeds = seed_sequences
        
        for iteration in range(1, num_iterations + 1):
            # First iteration: use provided seeds
            # Later iterations: use top performers from previous iteration
            results = self.run_iteration(
                iteration=iteration,
                seed_sequences=current_seeds,
                num_variants=90,
                num_random=(10 if iteration == 1 else 0)  # Include random sequences only in first iteration
            )
            
            # Extract top sequences for next iteration
            top_sequences = [result["sequence"] for result in results["results"][:top_n]]
            current_seeds = top_sequences
        
        # Compile final results
        top_hits = self.compile_top_hits(num_iterations, top_n=100)
        
        return top_hits
    
    def compile_top_hits(self, num_iterations, top_n=100):
        """Compile the top hits across all iterations."""
        self.log(f"Compiling top {top_n} hits across all {num_iterations} iterations")
        
        all_results = []
        
        # Collect results from all iterations
        for iteration in range(1, num_iterations + 1):
            result_file = os.path.join(self.output_dir, f"iteration_{iteration}", 
                                      "docking_results", "docking_results.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    results = json.load(f)
                    
                    # Add iteration info to each result
                    for result in results["results"]:
                        result["iteration"] = iteration
                    
                    all_results.extend(results["results"])
        
        # Sort by binding energy (lowest first)
        sorted_results = sorted(all_results, key=lambda x: x["binding_energy"])
        
        # Take top N
        top_hits = sorted_results[:top_n]
        
        # Save to file
        output_file = os.path.join(self.output_dir, "top_100_binding_sequences.json")
        with open(output_file, 'w') as f:
            json.dump({"top_hits": top_hits}, f, indent=2)
        
        # Print summary
        self.log(f"\nTop {len(top_hits)} binding sequences compiled and saved to {output_file}")
        self.log("\nTop 10 overall binding candidates:")
        for i, hit in enumerate(top_hits[:10]):
            self.log(f"{i+1}. {hit['seq_id']} (Iteration {hit['iteration']}): " 
                   f"{hit['sequence']} (Binding Energy: {hit['binding_energy']:.2f})")
        
        return top_hits

def main():
    """Main function to run the optimization process."""
    parser = argparse.ArgumentParser(description="Protein Binding Optimization")
    parser.add_argument("--target", required=True, help="Path to target protein PDB file (IL-8)")
    parser.add_argument("--output", default="optimization_results", help="Output directory")
    parser.add_argument("--seeds", default="AYGE", help="Comma-separated list of seed sequences")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--top_n", type=int, default=10, 
                       help="Number of top sequences to use for next iteration")
    parser.add_argument("--alphafold_path", help="Path to AlphaFold installation")
    
    args = parser.parse_args()
    
    # Split seed sequences
    seed_sequences = args.seeds.split(",")
    
    # Initialize the optimizer
    optimizer = ProteinBindingOptimizer(
        target_pdb=args.target,
        output_dir=args.output,
        alphafold_path=args.alphafold_path
    )
    
    # Run the optimization
    optimizer.run_optimization(
        seed_sequences=seed_sequences,
        num_iterations=args.iterations,
        top_n=args.top_n
    )

if __name__ == "__main__":
    main()
