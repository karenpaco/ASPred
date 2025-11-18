#!/usr/bin/env python

import argparse
import os
import glob
import sys
import pandas as pd
import torch

from peft import PeftModel, PeftConfig
from transformers import EsmTokenizer, EsmForSequenceClassification

# Optional: count LoRA tensors if safetensors is available
def count_lora_tensors(adapter_dir):
    try:
        from safetensors.torch import load_file
        fn = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.isfile(fn):
            return 0
        sd = load_file(fn)
        return sum("lora_" in k.lower() for k in sd.keys())
    except Exception:
        return -1  # unknown / skipped

def pick_seq_column(df, override=None):
    if override:
        if override in df.columns:
            return override
        raise KeyError(
            f"--seq_column '{override}' not found. Available columns: {list(df.columns)}"
        )
    for col in ("sequence_alignment_aa", "sequence_aa", "sequence"):
        if col in df.columns:
            return col
    raise KeyError(
        "No sequence column found. Tried: sequence_alignment_aa, sequence_aa, sequence"
    )

def predict(model, tokenizer, sequences, threshold=0.5, batch_size=32):
    all_logits, all_probs, all_preds = [], [], []
    model.eval()
    device = next(model.parameters()).device

    # Clean sequences (drop NA / empty)
    sequences = [s for s in sequences if isinstance(s, str) and len(s) > 0]

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            out = model(**tokens)
            logits = out.logits
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] > threshold).int()

        all_logits.extend(logits.detach().cpu().tolist())
        all_probs.extend(probs[:, 1].detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return all_logits, all_probs, all_preds

def run_one_csv(model, tokenizer, csv_path, threshold, output_dir, sample_size,
                sample_output, output_prefix, batch_size, seq_column_override):
    print(f"\n📂 Input file: {csv_path}")
    df = pd.read_csv(csv_path)

    seq_col = pick_seq_column(df, override=seq_column_override)

    print(f"   Using column: {seq_col} (n={len(df)})")

    # Optional sampling
    target_n = None
    if sample_size:
        n = len(df)
        target_n = (
            sample_size if n >= sample_size else
            100 if n >= 100 else
            50 if n >= 50 else
            n
        )
        print(f"✨ Sampling {target_n} rows (random_state=42)")
        df = df.sample(n=target_n, random_state=42).reset_index(drop=True)
        if sample_output:
            print(f"💾 Saving sampled data to: {sample_output}")
            os.makedirs(os.path.dirname(sample_output), exist_ok=True)
            df.to_csv(sample_output, index=False)

    sequences = df[seq_col].tolist()

    # Predict
    logits, probs, preds = predict(model, tokenizer, sequences, threshold=threshold, batch_size=batch_size)

    # Add outputs
    df["logit_class0"] = [x[0] for x in logits]
    df["logit_class1"] = [x[1] for x in logits]
    df["prob_class1"]  = probs
    df["predicted_label"] = preds

    # Output naming
    base = os.path.basename(csv_path)
    sample_suffix = f"_sampled_{target_n}" if target_n is not None else ""
    if output_prefix:
        basename = f"{output_prefix}{sample_suffix}__thresh{threshold}_predictions.csv"
    else:
        basename = base.replace(".csv", f"{sample_suffix}__thresh{threshold}_predictions.csv") if base.endswith(".csv") else f"{base}{sample_suffix}__thresh{threshold}_predictions.csv"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, basename)
    else:
        out_path = os.path.join(os.path.dirname(csv_path), basename)

    df.to_csv(out_path, index=False)
    print(f"🔍 Predicted label counts: {df['predicted_label'].value_counts().to_dict()}")
    print(f"✅ Saved predictions to: {out_path}")
    save_post_analysis(df, out_path, seq_col, threshold)

    return out_path
import matplotlib.pyplot as plt
import seaborn as sns

def save_post_analysis(df, out_path, seq_col, threshold):
    base_dir = os.path.dirname(out_path)
    base_name = os.path.splitext(os.path.basename(out_path))[0]

    # 1) Histogram + density plot of predicted probabilities
    plt.figure(figsize=(6,4))
    sns.histplot(df["prob_class1"], kde=True, bins=30, color="steelblue")
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
    plt.xlabel("Predicted probability (class 1)")
    plt.ylabel("Count")
    plt.title("Prediction probability distribution")
    plt.legend()
    plot_file = os.path.join(base_dir, f"{base_name}_probs.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"📊 Saved histogram/density plot to: {plot_file}")

    # 2) Write summary text file
    counts = df["predicted_label"].value_counts().to_dict()
    summary_file = os.path.join(base_dir, f"{base_name}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Total sequences: {len(df)}\n")
        f.write(f"Counts: {counts}\n")
    print(f"📝 Saved summary text to: {summary_file}")

    # 3) Save FASTA of positives
    pos_df = df[df["predicted_label"] == 1]
    fasta_file = os.path.join(base_dir, f"{base_name}_positives.fasta")
    with open(fasta_file, "w") as f:
        for i, row in pos_df.iterrows():
            seq_id = row.get("sequence_id", f"seq{i}")  # fallback if no ID
            seq = row[seq_col]
            f.write(f">{seq_id}\n{seq}\n")
    print(f"🧬 Saved positive sequences to FASTA: {fasta_file}")
    txt_file = os.path.join(base_dir, f"{base_name}_positives.txt")
    with open(txt_file, "w") as f:
        f.write(f"Threshold used: {threshold}\n")
        f.write(f"Total positives: {len(df[df['predicted_label'] == 1])}\n\n")
        for i, row in df[df["predicted_label"] == 1].iterrows():
            seq_id = row.get("sequence_id", f"seq{i}")  # fallback
            seq = row[seq_col]
            f.write(f">{seq_id}\n{seq}\n")
    print(f"📝 Saved positive sequences text file to: {txt_file}")

def main():
    ap = argparse.ArgumentParser(description="LoRA/PEFT ESM2 inference on single CSV or all *_processed.csv under a root")
    ap.add_argument("--model_path", required=True, help="Directory with adapter_model.safetensors and adapter_config.json")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input_csv", help="Single CSV to run")
    grp.add_argument("--input_root", help="Root directory; recursively find *_processed.csv")

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output_dir",type=str, default="inference", help="Directory to write outputs. Defaults to ./inference. "
         "If --input_root is used, the input folder structure is mirrored.")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sample_size", type=int, default=None, help="Optional sample size")
    ap.add_argument("--sample_output", type=str, default=None, help="Optional path to save the sampled CSV")
    ap.add_argument("--output_prefix", type=str, default=None, help="Optional base name for output CSV (no extension)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output CSV already exists")
    ap.add_argument("--sanity_checks", action="store_true", help="Print adapter status and quick LoRA checks")
    # after other ap.add_argument(...) lines
    ap.add_argument(
        "--seq_column",
        type=str,
        default=None,
        help="Name of the sequence column to use (overrides auto-detect). "
             "Common options: sequence, sequence_aa, sequence_alignment_aa"
    )

    args = ap.parse_args()

    # --- Load adapter config/base ---
    cfg = PeftConfig.from_pretrained(args.model_path)
    base_name = cfg.base_model_name_or_path
    print(f"Base model: {base_name}")

    # Optional peek at adapter_config.json fields
    try:
        import json
        cfgj = json.load(open(os.path.join(args.model_path, "adapter_config.json")))
        print("task_type:", cfgj.get("task_type"))
        print("modules_to_save:", cfgj.get("modules_to_save"))
    except Exception:
        pass

    # Count LoRA tensors
    n_lora = count_lora_tensors(args.model_path)
    if n_lora >= 0:
        print(f"adapter_model contains {n_lora} LoRA tensors")

    # Build model + tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = EsmForSequenceClassification.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base, args.model_path).to(device).eval()
    tokenizer = EsmTokenizer.from_pretrained(base_name)

    # Sanity check: show adapter status and difference ON vs OFF on a tiny batch
    if args.sanity_checks:
        try:
            print(getattr(model, "get_model_status")())
        except Exception:
            pass
        try:
            # Use a tiny synthetic batch if no CSV given yet
            test_seqs = ["ACDEFGHIKLMNPQRSTVWY"] * 4
            t = tokenizer(test_seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            t = {k: v.to(device) for k, v in t.items()}
            with torch.no_grad():
                on  = model(**t).logits.softmax(-1)[:,1]
                from contextlib import contextmanager
                # new PEFT has context manager disable_adapter
                try:
                    m = getattr(model, "disable_adapter")
                    cm = m() if callable(m) else None
                except Exception:
                    cm = None
                if cm is not None:
                    with cm:
                        off = model(**t).logits.softmax(-1)[:,1]
                else:
                    # fallback: temporarily disable then re-enable
                    try:
                        model.disable_adapters()
                        off = model(**t).logits.softmax(-1)[:,1]
                        model.enable_adapters()
                    except Exception:
                        off = on
                delta = float((on - off).abs().mean())
            print(f"mean|Δ|(adapter ON vs OFF) on dummy batch: {delta:.6f}")
        except Exception as e:
            print(f"(sanity_checks) skipped: {e}")

    # --- Build file list ---
    if args.input_csv:
        files = [args.input_csv]
    else:
        pattern = os.path.join(args.input_root, "**", "*_processed.csv")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            print(f"❌ No files matched: {pattern}", file=sys.stderr)
            sys.exit(2)
        print(f"Found {len(files)} CSVs under {args.input_root}")

    # --- Run all ---
    for csv_path in files:
        base = os.path.basename(csv_path)
        sample_suffix = f"_sampled_{args.sample_size}" if args.sample_size else ""
        if args.output_prefix:
            basename = f"{args.output_prefix}{sample_suffix}__thresh{args.threshold}_predictions.csv"
        else:
            basename = base.replace(".csv", f"{sample_suffix}__thresh{args.threshold}_predictions.csv") \
                       if base.endswith(".csv") else f"{base}{sample_suffix}__thresh{args.threshold}_predictions.csv"
    
        # NEW: decide the output directory
        if args.output_dir:
            if args.input_root:
                rel_dir = os.path.relpath(os.path.dirname(csv_path), start=args.input_root)
                out_dir = os.path.join(args.output_dir, rel_dir)   # mirror tree under output_dir
            else:
                out_dir = args.output_dir                          # single CSV -> drop into output_dir
        else:
            out_dir = os.path.dirname(csv_path)                    # (fallback) beside CSV
    
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, basename)
    
        if args.skip_existing and os.path.isfile(out_path):
            print(f"⚠️  Skipping existing: {out_path}")
            continue
    
        run_one_csv(
            model=model,
            tokenizer=tokenizer,
            csv_path=csv_path,
            threshold=args.threshold,
            output_dir=out_dir,          # pass the per-file target directory
            sample_size=args.sample_size,
            sample_output=args.sample_output,
            output_prefix=args.output_prefix,
            batch_size=args.batch_size,
            seq_column_override=args.seq_column,
        )



if __name__ == "__main__":
    main()
