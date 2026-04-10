"""
Transformer fine-tuning utilities for sentiment classification.
Configured for mDeBERTa-v3-base on the KazSAnDRA dataset.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
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
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}


class SentimentDataset(Dataset):
    """PyTorch dataset for sentiment classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
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


def get_training_args(output_dir, num_epochs=3, batch_size=16, learning_rate=1e-5,
                      weight_decay=1e-3, warmup_steps=800):
    """Training arguments matching KazSAnDRA paper setup."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
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


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss for imbalanced datasets."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def get_predictions(trainer, dataset):
    """Get predictions, probabilities, and labels from a dataset."""
    output = trainer.predict(dataset)
    logits = output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    labels = output.label_ids
    return preds, probs, labels
