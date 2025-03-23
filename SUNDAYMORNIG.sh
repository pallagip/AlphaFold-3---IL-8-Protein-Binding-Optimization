#!/bin/bash
#SBATCH -A c_ppkeitk
#SBATCH --job-name=alphafold_2025
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ================= Configuration =================
# File paths and basic configuration
RESULTS_DIR="$HOME/project/results/sundayevening"
IL8_BASENAME="IL8"
IL8_SEQ="MTSKLAVALLAAFLISAALCEGAVLPRSAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS"
NUM_ROUNDS=2
NUM_VARIANTS_PER_ROUND=2
NUM_TOP_HITS=4

# System paths
CONDA_ENV="parafold"
PARALLEL_FOLD_PATH="/opt/software/packages/alphafold/ParallelFold"
DB_PATH="/opt/software/packages/alphafold/alphafold-db"

# ================= Setup Environment =================
echo "[INFO] Setting up environment"
source /opt/software/packages/miniconda/3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH=${PARALLEL_FOLD_PATH}/nvidialib/:$LD_LIBRARY_PATH

# Create results directory
mkdir -p ${RESULTS_DIR}

# ================= Helper Functions =================

# Run monomer alignment
run_monomer_alignment() {
    local fasta_path=$1
    local output_dir=$2
    local variant_name=$3
    
    echo "[INFO] Running sequence alignment for: ${variant_name}"
    cd ${PARALLEL_FOLD_PATH}/scripts
    ./prep_features.sh ${fasta_path} ${output_dir} ${DB_PATH}
    return $?
}

# Run monomer prediction
run_monomer_prediction() {
    local fasta_path=$1
    local output_dir=$2
    local variant_name=$3
    
    echo "[INFO] Running structure prediction for: ${variant_name}"
    cd ${PARALLEL_FOLD_PATH}
    ./run_alphafold.sh -d ${DB_PATH} -o ${output_dir} -p monomer_ptm -i ${fasta_path} -t 2022-01-01 \
       -m model_1,model_2,model_3,model_4,model_5 -r none
    return $?
}

# Run multimer prediction
run_multimer_prediction() {
    local fasta_path=$1
    local output_dir=$2
    local variant_name=$3
    
    echo "[INFO] Running multimer prediction for: ${variant_name}"
    cd ${PARALLEL_FOLD_PATH}
    ./run_alphafold.sh -d ${DB_PATH} -o ${output_dir} -p multimer -i ${fasta_path} -t 2022-01-01 \
       -m model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3,model_4_multimer_v3,model_5_multimer_v3 -r none
    return $?
}

# Function: calc_distance
# Calculate the Euclidean distance between two atoms given their coordinates.
calc_distance() {
    awk -v x1="$1" -v y1="$2" -v z1="$3" \
        -v x2="$4" -v y2="$5" -v z2="$6" \
        'BEGIN { printf "%.2f", sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2) }'
}

# Function: calc_surface_area
# Assign approximate atomic radii and compute surface area = 4*pi*r^2
calc_surface_area() {
    local elem="$1"
    local pi=3.14159
    local r=1.5
    if [ "$elem" = "N" ]; then
        r=1.65
    elif [ "$elem" = "C" ]; then
        r=1.76
    elif [ "$elem" = "O" ]; then
        r=1.40
    elif [ "$elem" = "S" ]; then
        r=1.85
    fi
    echo "$(echo "4 * $pi * $r * $r" | bc -l)"
}

# Function: calc_binding_energy
# Encapsulates the PDB parsing and approximate energy calculations.
calc_binding_energy() {
    local pdb_file=$1
    local variant_name=$2  # not strictly used, but retained for clarity
    
    # Create a temporary working directory
    local tempdir=$(mktemp -d)
    
    # Step 1. Get the unique chain IDs from ATOM lines and choose the first two.
    local chains=$(awk '/^ATOM/ { c=substr($0,22,1); print c }' "$pdb_file" | sort | uniq)
    local chainA=$(echo "$chains" | sed -n '1p')
    local chainB=$(echo "$chains" | sed -n '2p')

    if [ -z "$chainA" ] || [ -z "$chainB" ]; then
        echo "[ERROR] The PDB file must contain at least two chains."
        rm -r "$tempdir"
        echo "999999"  # Return a large "error" energy so it won't rank
        return 1
    fi

    # Step 2. Extract coordinates and elements for each chain.
    awk -v chain="$chainA" '
        substr($0,22,1)==chain && /^ATOM/ {
            x = substr($0,31,8)+0;
            y = substr($0,39,8)+0;
            z = substr($0,47,8)+0;
            element = substr($0,77,2);
            gsub(/ /, "", element);
            print x, y, z, element
        }
    ' "$pdb_file" > "$tempdir/chainA.dat"

    awk -v chain="$chainB" '
        substr($0,22,1)==chain && /^ATOM/ {
            x = substr($0,31,8)+0;
            y = substr($0,39,8)+0;
            z = substr($0,47,8)+0;
            element = substr($0,77,2);
            gsub(/ /, "", element);
            print x, y, z, element
        }
    ' "$pdb_file" > "$tempdir/chainB.dat"

    # Define constants for energy calculation
    local CONTACT_ENERGY=-0.15    # kcal/mol if atoms are within 5.0 Å
    local HBOND_ENERGY=-1.5       # additional if donor/acceptor (N/O) within 3.5 Å
    local ELECTROSTATIC_WEIGHT=0.2
    local COULOMB_CONST=332.059   # kcal·Å/(mol·e^2)
    local EPSILON_R=4.0           # relative dielectric constant
    local SOLVATION_FACTOR=0.1
    local pi=3.14159

    local contact_sum=0
    local hbond_sum=0
    local electrostatic_sum=0
    local solvation_sum=0

    # Step 4. Loop over all pairs of atoms (chain A vs chain B).
    while read x1 y1 z1 elem1; do
        while read x2 y2 z2 elem2; do
            dist=$(calc_distance "$x1" "$y1" "$z1" "$x2" "$y2" "$z2")
            # Contact if distance <= 5.0 Å
            is_contact=$(echo "$dist <= 5.0" | bc -l)
            if [ "$is_contact" -eq 1 ]; then
                contact_sum=$(echo "$contact_sum + $CONTACT_ENERGY" | bc -l)
                
                # If both atoms are N or O, evaluate hydrogen bonding and electrostatics.
                if [[ "$elem1" =~ ^(N|O)$ ]] && [[ "$elem2" =~ ^(N|O)$ ]]; then
                    is_hbond=$(echo "$dist <= 3.5" | bc -l)
                    if [ "$is_hbond" -eq 1 ]; then
                        hbond_sum=$(echo "$hbond_sum + $HBOND_ENERGY" | bc -l)
                    fi
                    # Simple partial charges: N: +0.5, O: -0.5
                    q1=0
                    q2=0
                    if [ "$elem1" = "N" ]; then q1=0.5; elif [ "$elem1" = "O" ]; then q1=-0.5; fi
                    if [ "$elem2" = "N" ]; then q2=0.5; elif [ "$elem2" = "O" ]; then q2=-0.5; fi
                    
                    # Electrostatic term: weight * (coulomb_const * q1*q2)/(epsilon_r * dist)
                    elec_term=$(echo "$ELECTROSTATIC_WEIGHT * ($COULOMB_CONST * ($q1 * $q2)) / ($EPSILON_R * $dist)" | bc -l)
                    electrostatic_sum=$(echo "$electrostatic_sum + $elec_term" | bc -l)
                fi
            fi
        done < "$tempdir/chainB.dat"
        # Rewind chainB file after each pass
        exec 3<"$tempdir/chainB.dat"
    done < "$tempdir/chainA.dat"

    # Step 5. Calculate approximate solvation energy for all atoms in both chains.
    for file in "$tempdir/chainA.dat" "$tempdir/chainB.dat"; do
        while read _ _ _ elem; do
            area=$(calc_surface_area "$elem")
            solvation_sum=$(echo "$solvation_sum + ($SOLVATION_FACTOR * $area)" | bc -l)
        done < "$file"
    done

    # Step 6. Sum up all energy contributions.
    local total_energy
    total_energy=$(echo "$contact_sum + $hbond_sum + $electrostatic_sum + $solvation_sum" | bc -l)

    # Clean up
    rm -r "$tempdir"

    # Return the numeric value (so we can capture it in a variable)
    echo "$total_energy"
}

# Generate a random mutation of a sequence
mutate_sequence() {
    local base_seq=$1
    local amino_acids="ACDEFGHIKLMNPQRSTVWY"
    local seq_chars=( $(echo $base_seq | grep -o .) )
    local num_mutations=$((1 + RANDOM % 3))
    
    for ((i=0; i<num_mutations; i++)); do
        local pos=$((RANDOM % ${#seq_chars[@]}))
        local new_aa=${amino_acids:$((RANDOM % ${#amino_acids})):1}
        seq_chars[$pos]=$new_aa
    done
    
    # Print the mutated sequence as a single line
    echo "${seq_chars[@]}" | sed 's/ //g'
}

# ================= Main Pipeline =================
echo "[INFO] Starting protein structure prediction pipeline"

# Step 1: Fold IL-8 structure
IL8_DIR="${RESULTS_DIR}/IL8_monomer"
mkdir -p ${IL8_DIR}
IL8_FASTA="${IL8_DIR}/${IL8_BASENAME}.fasta"

echo ">${IL8_BASENAME}" > ${IL8_FASTA}
echo "${IL8_SEQ}" >> ${IL8_FASTA}

echo "[INFO] Folding IL-8 structure"
run_monomer_alignment ${IL8_FASTA} ${IL8_DIR} ${IL8_BASENAME}
run_monomer_prediction ${IL8_FASTA} ${IL8_DIR} ${IL8_BASENAME}

IL8_PDB_PATH="${IL8_DIR}/ranked_0.pdb"
if [ ! -f ${IL8_PDB_PATH} ]; then
    echo "[ERROR] Failed to generate IL-8 structure"
    exit 1
fi

# Initialize with a starting sequence (example)
BASE_SEQUENCES=("ACDEFGHIKLMNPQRSTVWYVSRTFEDQMPYLIWKHNAG")
TOP_SEQUENCES=("${BASE_SEQUENCES[@]}")
ALL_SCORED_VARIANTS=()

# Run multiple rounds of variant generation and evaluation
for ((round=0; round<${NUM_ROUNDS}; round++)); do
    echo "[INFO] Starting Round $((round+1)) of ${NUM_ROUNDS}"
    
    ROUND_SCORED_VARIANTS=()
    
    # Generate variants for this round
    for base_seq in "${TOP_SEQUENCES[@]}"; do
        for ((i=0; i<${NUM_VARIANTS_PER_ROUND}; i++)); do
            variant_name="var_r${round}_${i}"
            variant_seq=$(mutate_sequence "${base_seq}")
            
            echo "[INFO] Processing variant: ${variant_name}"
            
            # Create variant directory and FASTA
            variant_dir="${RESULTS_DIR}/${variant_name}_monomer"
            mkdir -p ${variant_dir}
            variant_fasta="${variant_dir}/${variant_name}.fasta"
            
            echo ">${variant_name}" > ${variant_fasta}
            echo "${variant_seq}" >> ${variant_fasta}
            
            # Run monomer prediction
            run_monomer_alignment ${variant_fasta} ${variant_dir} ${variant_name}
            run_monomer_prediction ${variant_fasta} ${variant_dir} ${variant_name}
            
            # Create multimer directory and FASTA
            multimer_dir="${RESULTS_DIR}/${variant_name}_multimer"
            mkdir -p ${multimer_dir}
            multimer_fasta="${multimer_dir}/${variant_name}_multimer.fasta"
            
            echo ">${variant_name}" > ${multimer_fasta}
            echo "${variant_seq}" >> ${multimer_fasta}
            echo ">${IL8_BASENAME}" >> ${multimer_fasta}
            echo "${IL8_SEQ}" >> ${multimer_fasta}
            
            # Run multimer prediction
            run_multimer_prediction ${multimer_fasta} ${multimer_dir} ${variant_name}
            
            # Calculate binding energy
            complex_pdb="${multimer_dir}/ranked_0.pdb"
            if [ -f ${complex_pdb} ]; then
                binding_energy=$(calc_binding_energy ${complex_pdb} ${variant_name})
                echo "[INFO] Variant ${variant_name} binding energy: ${binding_energy} kcal/mol"
                ROUND_SCORED_VARIANTS+=("${variant_name}:${variant_seq}:${binding_energy}")
            else
                echo "[WARNING] Failed to generate complex structure for ${variant_name}"
            fi
        done
    done
    
    # Sort variants by binding energy and get top hits
    echo "[INFO] Finding top hits for Round $((round+1))"
    
    # Add all variants from this round to overall list
    ALL_SCORED_VARIANTS+=("${ROUND_SCORED_VARIANTS[@]}")
    
    # Sort variants by numeric value in field 3 (binding energy)
    SORTED_VARIANTS=($(printf '%s\n' "${ROUND_SCORED_VARIANTS[@]}" | sort -t: -k3 -n))
    
    # Get top N hits for next round
    TOP_SEQUENCES=()
    echo "[INFO] Top ${NUM_TOP_HITS} hits for Round $((round+1)):"
    for ((i=0; i<${NUM_TOP_HITS} && i<${#SORTED_VARIANTS[@]}; i++)); do
        IFS=':' read -r name seq energy <<< "${SORTED_VARIANTS[i]}"
        echo "  $((i+1)). ${name} | Binding Energy = ${energy} kcal/mol | Seq = ${seq}"
        TOP_SEQUENCES+=("${seq}")
    done
done

# Report overall top hits
echo "[INFO] All rounds completed. Finding overall top hits..."
SORTED_ALL=($(printf '%s\n' "${ALL_SCORED_VARIANTS[@]}" | sort -t: -k3 -n))

echo "[INFO] Top 10 variants across all rounds (lowest energy first):"
for ((i=0; i<10 && i<${#SORTED_ALL[@]}; i++)); do
    IFS=':' read -r name seq energy <<< "${SORTED_ALL[i]}"
    echo "  $((i+1)). ${name} | Energy = ${energy} kcal/mol | Seq = ${seq}"
done

echo "[INFO] Done! Results are available in ${RESULTS_DIR}"