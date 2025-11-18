#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import re

# ========= CONFIG =========
CSV_FILE = "merged_log_wandb_metrics_final_end.csv"
OUTPUT_DIR = "analysis_metrics_best_models_loss"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= LOAD =========
df = pd.read_csv(CSV_FILE)

# ========= Drop rows with missing avg_val_loss =========
df = df[df["avg_val_loss"].notna()].reset_index(drop=True)

# ========= Identify best trial per file =========
best_trials = (
    df.loc[df.groupby("file")["avg_val_loss"].idxmin()]
    .reset_index(drop=True)
)

# ========= Aggregate =========
records = []
for _, best_trial_row in best_trials.iterrows():
    file = best_trial_row["file"]
    trial = best_trial_row["trial"]
    model_type = "8M" if "8M" in file else "650M"
    objective = "Loss"

    # Clean label
    m = re.search(r"optuna_(?:loss_)?(?:8M_)?([A-Za-z]+)_([A-Za-z]+|stacked)", file)
    if m:
        prefix = m.group(1).upper()
        suffix = "Heavy-Light" if m.group(2) == "stacked" else m.group(2).capitalize()
        model = f"{prefix}_{suffix}"
    else:
        model = file

    # All rows for this trial
    trial_rows = df[(df["file"] == file) & (df["trial"] == trial)].copy()
    best_epoch_rows = []
    for fold, fold_rows in trial_rows.groupby("fold"):
        idx = fold_rows["eval_loss"].idxmin()
        best_epoch_rows.append(fold_rows.loc[idx])

    best_epoch_df = pd.DataFrame(best_epoch_rows)

    acc = best_epoch_df["eval_accuracy"].mean()
    prec = best_epoch_df["eval_precision"].mean()
    rec = best_epoch_df["eval_recall"].mean()

    auc_cols = ["fold0_auc", "fold1_auc", "fold2_auc"]
    auc_vals = best_trial_row[auc_cols]
    avg_auc = np.nanmean(auc_vals)

    records.append({
        "Model": model,
        "ModelType": model_type,
        "Objective": objective,
        "Loss": best_trial_row["avg_val_loss"],
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "AUROC": avg_auc
    })

df_out = pd.DataFrame(records)
df_out.to_csv(os.path.join(OUTPUT_DIR, "best_models_loss.csv"), index=False)
print("✅ Done: Loss-optimized models saved.")
