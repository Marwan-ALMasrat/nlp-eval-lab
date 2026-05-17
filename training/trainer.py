"""
Module 7 Week A — Fine-Tune DistilBERT for App-Review Sentiment.

Default run: `python training/trainer.py` reads `data/app_reviews_train.csv`
(7,472 reviews across 9 apps with 3 sentiment classes: 0=negative, 1=neutral,
2=positive) and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.
    Returns a DatasetDict with "train" and "test" keys.
    """
    df = pd.read_csv(data_path)
    df["label"] = df["label"].astype(int)
    try:
        ds = Dataset.from_pandas(df, preserve_index=False)
    except TypeError:
        ds = Dataset.from_dict(df.to_dict(orient="list"))
    return ds.train_test_split(test_size=test_size, seed=seed)


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.
    Use truncation=True and max_length=max_length.
    Padding is applied dynamically by DataCollatorWithPadding at training time.
    """
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return TrainingArguments configured for fine-tuning."""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=seed,
        load_best_model_at_end=True,
    )


def compute_metrics(eval_pred):
    """Convert (logits, labels) into accuracy and macro_f1."""
    logits, labels = eval_pred
    preds    = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """Construct and train a Trainer. Returns the trained Trainer."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.
    Returns accuracy, macro_f1, per_class_f1, per_class_precision, per_class_recall.
    """
    id2label    = trainer.model.config.id2label
    preds_output = trainer.predict(tokenized_test)
    logits       = preds_output.predictions
    labels       = preds_output.label_ids
    pred_idx     = np.argmax(logits, axis=1)

    return {
        "accuracy": float(accuracy_score(labels, pred_idx)),
        "macro_f1": float(f1_score(labels, pred_idx, average="macro")),
        "per_class_f1":       {id2label[i]: float(v) for i, v in enumerate(f1_score(labels, pred_idx, average=None))},
        "per_class_precision": {id2label[i]: float(v) for i, v in enumerate(precision_score(labels, pred_idx, average=None, zero_division=0))},
        "per_class_recall":    {id2label[i]: float(v) for i, v in enumerate(recall_score(labels, pred_idx, average=None, zero_division=0))},
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> None:
    """Orchestrate the full training pipeline."""
    data_path  = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    # 1. Data
    ds        = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 2. Train
    training_args = make_training_args(output_dir)
    trainer       = train_classifier(tokenized, model_name, training_args, tokenizer)

    # 3. Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 4. Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}")

    # 5. Predictions CSV
    pred_output = trainer.predict(tokenized["test"])
    pred_idx    = np.argmax(pred_output.predictions, axis=1)
    pred_probs  = _softmax(pred_output.predictions)
    id2label    = trainer.model.config.id2label

    df_out = pd.DataFrame({
        "text":                  ds["test"]["text"],
        "label":                 [id2label[i] for i in ds["test"]["label"]],
        "predicted_label":       [id2label[i] for i in pred_idx],
        "predicted_probability": [float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    })
    for idx, name in id2label.items():
        df_out[f"prob_{name}"] = [float(pred_probs[i, idx]) for i in range(len(pred_idx))]
    df_out.to_csv("predictions.csv", index=False)

    # 6. Confusion matrix
    label_names = list(id2label.values())
    cm    = confusion_matrix([id2label[i] for i in ds["test"]["label"]],
                              [id2label[i] for i in pred_idx], labels=label_names)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv("confusion_matrix.csv")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    # 7. Push to Hugging Face Hub (skipped in CI)
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/<your-username>/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` and try again.")


if __name__ == "__main__":
    main()