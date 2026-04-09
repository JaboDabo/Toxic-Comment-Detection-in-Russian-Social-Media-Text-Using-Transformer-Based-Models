"""
Transformer fine-tuning utilities for toxic comment classification.
Supports ruBERT and XLM-RoBERTa.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Monkey-patch transformers to skip TF entirely (fixes Keras 3 conflict)
import transformers.utils.import_utils as _tf_check
_tf_check._tf_available = False

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

RANDOM_SEED = 42
LABEL2ID = {"non-toxic": 0, "toxic": 1}
ID2LABEL = {0: "non-toxic", 1: "toxic"}


class ToxicDataset(Dataset):
    """PyTorch dataset for toxic comment classification."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_splits(data_dir):
    """Load train/val/test CSVs and return texts + labels."""
    splits = {}
    for name in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
        texts = df["clean_comment"].fillna("").tolist()
        labels = df["toxic"].tolist()
        splits[name] = (texts, labels)
    return splits


def compute_metrics_hf(eval_pred):
    """Compute metrics compatible with HuggingFace Trainer."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "roc_auc": auc,
    }


def build_model_and_tokenizer(model_name, num_labels=2):
    """Load pre-trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tokenizer


def get_training_args(output_dir, num_epochs=3, batch_size=8, learning_rate=2e-5,
                      gradient_accumulation_steps=2):
    """Standard training arguments for transformer fine-tuning."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )


def get_predictions(trainer, dataset):
    """Get predictions, probabilities, and labels from a dataset."""
    output = trainer.predict(dataset)
    logits = output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    labels = output.label_ids
    return preds, probs, labels
