# ASPred (Antigen Specificity Predictor)

ASPred is a protein language–model–based classifier that predicts antigen-specific
B cell receptors (BCRs) directly from amino acid sequences.  
It supports multiple chain formats (heavy, light, and heavy+light stacked) and 
multiple viral antigens (Influenza HA, HIV gp120, SARS-CoV-2 RBD).

ASPred fine-tunes ESM-2 transformer models using LoRA/PEFT and Optuna-based hyperparameter 
optimization, producing compact and high-performance antigen classifiers that are fast to train 
and easy to deploy.

This repository includes:
- training code for LoRA-based fine-tuning  
- dataset structure (train/test splits per antigen × chain type)  
- inference pipeline with batch prediction & plotting  
- SLURM template generator for large-scale HPC training  
- configuration files, notebooks, and example results  

This repository is intended for interview review, reproducibility, and future peer-review.

---

## Features

- **LoRA fine-tuning** on ESM-2 8M or ESM-2 650M  
- **Optuna hyperparameter search** (learning rate, batch size, LoRA ranks, dropout, etc.)  
- **Multi-antigen** (FLU, HIV, SARS-CoV-2)  
- **Multi-chain** (heavy, light, stacked heavy+light)  
- **Metrics:** F1, accuracy, ROC-AUC, loss curves  
- **Inference pipeline** with probability histograms, FASTA export, and sampling  
- **SLURM job generator** for running cross-antigen fine-tuning at scale  

---

# Dataset Format

Each dataset folder corresponds to a particular **antigen** and **BCR chain type**:

- FLU (Influenza)
- HIV
- SARS (SARS-CoV-2 RBD)

Each dataset folder contains:
train.csv
test.csv


Each CSV uses the schema:

| column        | description                               |
|---------------|-------------------------------------------|
| sequence_id   | unique identifier                         |
| sequence_aa   | amino acid sequence                       |
| label         | 1 = antigen-specific, 0 = not specific    |

Example:
ID,sequence,label
029_09_2A06,EVQLVE...WGQGTLVTVSS,1
AFL34761_AFL34760,QSVEES...WGQGTLVTVSS,0

# Model fine tuning

Finetuning is handled by:

src/optimize_f1_early_stop.py

#This script:
- loads ESM-2 model  
- applies LoRA adapters  
- runs Optuna hyperparameter search  
- tracks metrics and early stopping  
- saves best models and metrics  

### Example (local, CPU/GPU):

```bash
python src/optimize_f1_early_stop.py \
    --dataset_name FLU_heavy \
    --target_metric f1 \
    --model_name esm2_8M \
    --hf_model_name facebook/esm2_t6_8M_UR50D \
    --base_dir 
    
#For large-scale training, use:

scripts/slurm_template_generator.sh


#This script scans datasets/ and produces one SLURM job per dataset folder.

Generate all SLURM jobs:
bash scripts/slurm_template_generator.sh

Submit a generated job:
sbatch scripts/train_FLU_heavy.sh

##Inference is performed with:

src/inference.py



