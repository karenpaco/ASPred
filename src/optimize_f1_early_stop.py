import pandas as pd
import argparse
import os
import joblib
import datetime
import torch
import wandb
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve, roc_auc_score
)

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    EsmTokenizer, EsmForSequenceClassification,
    Trainer, TrainingArguments, EvalPrediction,
    EarlyStoppingCallback, set_seed
)
from optuna.visualization import plot_optimization_history, plot_param_importances

# ========= UTILS =========
def tokenize(batch, tokenizer):
    return tokenizer(batch["sequence"], truncation=True, padding="max_length", max_length=512)

def compute_metrics(pred: EvalPrediction):
    preds = torch.argmax(torch.tensor(pred.predictions), dim=1)
    labels = torch.tensor(pred.label_ids)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0)
    }

def plot_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()

# ========= MODEL INIT =========
def model_init(trial, model_name):
    model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=2)
    lora_config = LoraConfig(
        r=trial.suggest_categorical("lora_r", [4, 8]),
        lora_alpha=trial.suggest_int("lora_alpha", 16, 64),
        target_modules=["query", "key", "value"],
        lora_dropout=trial.suggest_float("lora_dropout", 0.05, 0.1),
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    return get_peft_model(model, lora_config)

# ========= OBJECTIVE =========
def objective(trial, args, df, tokenizer):
    set_seed(42)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    lr = trial.suggest_float("learning_rate", 5e-7, 2e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    num_epochs = trial.suggest_int("epochs", 4, 8)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)

    wandb.init(
        project=f"{args.model_name}_{args.target_metric}_{args.dataset_name}_cv",
        name=f"{args.model_name}_{args.target_metric}_{args.dataset_name}_trial{trial.number}_{timestamp}",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
            "n_folds": args.n_folds
        },
        resume="allow"
    )

    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n========== Fold {fold+1} ==========")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        train_ds = Dataset.from_pandas(df_train).map(lambda x: tokenize(x, tokenizer), batched=True)
        val_ds = Dataset.from_pandas(df_val).map(lambda x: tokenize(x, tokenizer), batched=True)

        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        model_id = f"{args.model_name}_{args.target_metric}_{args.dataset_name}_fold{fold}_trial{trial.number}_{timestamp}"
        fold_dir = os.path.join(args.model_dir, model_id)
        os.makedirs(fold_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            logging_steps=10,
            report_to="none",
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            warmup_ratio=warmup_ratio,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False
        )

        trainer = Trainer(
            model_init=lambda: model_init(trial, args.hf_model_name),
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )

        trainer.train()
        trainer.model.save_pretrained(fold_dir)

        preds = trainer.predict(val_ds)
        probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
        y_pred = torch.argmax(torch.tensor(preds.predictions), dim=1).numpy()
        y_true = torch.tensor(preds.label_ids).numpy()

        pd.DataFrame({
            "label": y_true, "pred": y_pred, "prob": probs
        }).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        # Plotting
        cm = confusion_matrix(y_true, y_pred)
        cm_path = os.path.join(args.plots_dir, f"{model_id}_cm.png")
        plot_confusion_matrix(cm, ["Negative", "Positive"], cm_path)
        wandb.log({f"fold{fold}_confusion_matrix": wandb.Image(cm_path)})

        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        roc_path = os.path.join(args.plots_dir, f"{model_id}_roc.png")
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC={auc:.2f}", color="darkorange")
        plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("ROC Curve"); plt.legend()
        plt.savefig(roc_path); plt.close()
        wandb.log({f"fold{fold}_roc": wandb.Image(roc_path), f"fold{fold}_auc": auc})

        f1 = f1_score(y_true, y_pred, zero_division=0)
        fold_f1s.append(f1)
        wandb.log({f"fold{fold}_f1": f1})

    avg_f1 = sum(fold_f1s) / args.n_folds
    wandb.log({"avg_f1": avg_f1})
    wandb.finish()
    return avg_f1




# ========= MAIN =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="sars_heavy / sars_light / sars_stacked")
    parser.add_argument("--target_metric", type=str, default="f1", help="f1 / accuracy")
    parser.add_argument("--model_name", type=str, default="esm2_650M", help="esm2_650M, etc.")
    parser.add_argument("--hf_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--base_dir", type=str, default=".")

    args = parser.parse_args()

    # Paths
    args.dataset_path = os.path.join(args.base_dir, "datasets", args.dataset_name, "train.csv")
    args.output_dir = os.path.join(args.base_dir, "outputs")
    args.model_dir = os.path.join(args.output_dir, "models_f1")
    args.plots_dir = os.path.join(args.output_dir, "plots_f1")
    args.study_dir = os.path.join(args.output_dir, "studies_f1")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.study_dir, exist_ok=True)

    df = pd.read_csv(args.dataset_path)
    tokenizer = EsmTokenizer.from_pretrained(args.hf_model_name)

    study_name = f"optuna_study_{args.model_name}_{args.target_metric}_{args.dataset_name}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda trial: objective(trial, args, df, tokenizer), n_trials=20)

    # Save
    study_path = os.path.join(args.study_dir, f"{study_name}.pkl")
    joblib.dump(study, study_path)

    print("Best trial:")
    print(f"  Value ({args.target_metric}): {study.best_trial.value}")
    for key, val in study.best_trial.params.items():
        print(f"    {key}: {val}")

    # Visuals
    plot_optimization_history(study).write_image(os.path.join(args.plots_dir, f"optimization_history_{study_name}.png"))
    plot_param_importances(study).write_image(os.path.join(args.plots_dir, f"param_importance_{study_name}.png"))
    
    
    # ========== FINAL RETRAIN ==========
    print("\nRetraining final model with best hyperparameters on full dataset...")
    best_params = study.best_trial.params
    model = model_init(study.best_trial, args.hf_model_name)

    # Tokenize full dataset
    full_ds = Dataset.from_pandas(df).map(lambda x: tokenize(x, tokenizer), batched=True)
    full_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # TrainingArgs for full retrain
    final_dir = os.path.join(args.model_dir, f"best_model_{args.model_name}_{args.target_metric}_{args.dataset_name}")
    os.makedirs(final_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=final_dir,
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=best_params["batch_size"],
        num_train_epochs=best_params["epochs"],
        warmup_ratio=best_params["warmup_ratio"],
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to="wandb",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(final_dir)
    print(f"✅ Best model saved to: {final_dir}")