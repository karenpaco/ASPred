#!/bin/bash
#
# SLURM job generator for ASPred
# This template automatically creates one SLURM job per dataset folder.
# 
# NOTE:
# - This version uses RELATIVE PATHS and PLACEHOLDERS.
# - Edit paths and SLURM directives based on your cluster.
# - This script is cluster-agnostic and safe to publish.

# ========= CONFIGURATION ==========
ASPREDDIR="."                       # Root of the ASPred repo
DATASET_ROOT="$ASPREDDIR/datasets"  # Each dataset has train.csv/test.csv
SCRIPT_PATH="$ASPREDDIR/src/optimize_f1_early_stop.py"

MODEL_NAME="esm2_8M"
HF_MODEL="facebook/esm2_t6_8M_UR50D"

# ========= SLURM SETTINGS (EDIT FOR YOUR CLUSTER) =========
SLURM_ACCOUNT="<your_slurm_account>"    # Replace with your account
SLURM_PARTITION="gpu"                   # GPU partition name
SLURM_CPUS=8                            # Adjust for your environment
SLURM_GPUS=1                            # Number of GPUs
SLURM_MEM="32G"                         # Memory request
SLURM_TIME="24:00:00"                   # Job time limit

mkdir -p scripts logs

# ========= GENERATE JOB SCRIPTS ==========
for dir in "$DATASET_ROOT"/*/; do
    dir_name=$(basename "$dir")
    train_path="${dir}/train.csv"
    test_path="${dir}/test.csv"

    if [[ -f "$train_path" && -f "$test_path" ]]; then

        outfile="scripts/train_${dir_name}.sh"

        cat > "$outfile" << EOF
#!/bin/bash
#SBATCH --job-name=asprd_${dir_name}
#SBATCH --output=logs/asprd_${dir_name}_%j.out
#SBATCH --error=logs/asprd_${dir_name}_%j.err
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --gpus-per-task=$SLURM_GPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME
# #SBATCH --account=$SLURM_ACCOUNT    # Uncomment on clusters requiring accounts

# ========= ENV SETUP =========
source activate aspred_env  # Expects user to create an environment
# OR: conda activate esm2
# OR: module load python/pytorch

# ========= RUN TRAINING =========
python $SCRIPT_PATH \\
    --dataset_name $dir_name \\
    --target_metric f1 \\
    --model_name $MODEL_NAME \\
    --hf_model_name $HF_MODEL \\
    --base_dir $ASPREDDIR

EOF

        echo "Generated SLURM job: $outfile"

    else
        echo "Skipping $dir_name: missing train.csv/test.csv"
    fi
done
