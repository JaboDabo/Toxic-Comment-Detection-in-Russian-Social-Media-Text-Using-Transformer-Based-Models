# Sentiment Analysis of Kazakh Customer Reviews

## Project Overview

Binary polarity classification (positive vs. negative) of Kazakh-language customer reviews. Compares **4 models**: three classical TF-IDF baselines (Multinomial Naive Bayes, Logistic Regression, Linear SVM) and one transformer (**mDeBERTa-v3-base**). The paper positions itself as the **first evaluation of classical ML baselines and mDeBERTa-v3 on the KazSAnDRA benchmark**, filling gaps left by the original paper (Yeshpanov & Varol, LREC-COLING 2024). Research paper formatted with Springer Nature LaTeX template.

**Research question:** Can mDeBERTa-v3 beat the published KazSAnDRA SOTA (F1 = 0.81)?

## Dataset: KazSAnDRA

- **Source:** IS2AI / Nazarbayev University
- **Paper:** Yeshpanov & Varol (2024), LREC-COLING 2024
- **Access:** `load_dataset("issai/kazsandra", "polarity_classification")` on HuggingFace
- **Size:** ~180,064 Kazakh reviews
- **Domains:** Appstore, Bookstore, Mapping services
- **Variant:** `ib` (imbalanced) — matches the paper's best-reported setup
- **Splits:** Pre-split 80/10/10 (train/validation/test)
- **Labels:** 0 = negative, 1 = positive (derived from 1-2 vs 4-5 star ratings; 3-star excluded)
- **Columns:** `custom_id`, `text`, `text_cleaned` (use this), `label`, `domain`
- **Language quirks:** Code-switching — pure Kazakh Cyrillic, Kazakh Latin, mixed Cyrillic/Latin, mixed Russian/Kazakh

## Project Structure

```
midterm/
├── data/                         # HuggingFace datasets cache (gitignored)
├── notebooks/
│   ├── 01_eda_kazsandra.ipynb    # Exploratory data analysis
│   ├── 02_baselines.ipynb        # TF-IDF + MNB/LogReg/SVM baselines
│   ├── 03_mdeberta.ipynb         # mDeBERTa-v3 fine-tuning
│   └── 04_error_analysis.ipynb   # (planned) Error analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # KazSAnDRA HF loader → pandas DataFrames
│   ├── preprocessing.py          # Minimal text normalization
│   ├── baselines.py              # TF-IDF + MNB/LogReg/SVM pipeline builders
│   ├── evaluation.py             # Shared metrics, plots, save utilities
│   └── transformers_train.py     # mDeBERTa dataset, training, prediction utils
├── models/                       # Saved model checkpoints (gitignored)
├── results/
│   ├── figures/                  # EPS files for LaTeX
│   ├── metrics.json
│   └── test_predictions.csv
├── paper/                        # LaTeX paper (Springer Nature template)
├── requirements.txt
├── .gitignore
└── CLAUDE.md                     # This file
```

## How to Run (Step by Step)

### 1. Setup

```bash
pip install -r requirements.txt
```

Required: Python 3.10+, pandas, numpy, matplotlib, seaborn, scikit-learn, transformers, torch, datasets, accelerate, pyarrow.

### 2. Verify Dataset Access

```bash
python3 src/data_loader.py
```

Downloads KazSAnDRA from HuggingFace and prints split sizes + label distributions. First run will download ~100MB.

### 3. EDA — `notebooks/01_eda_kazsandra.ipynb`

Exploratory data analysis. Run all cells. Produces:
- Dataset size & splits summary
- Label distribution (positive vs negative)
- Domain distribution (Appstore, Bookstore, Mapping)
- Text length histograms by class
- Domain x label cross-tab
- Sample reviews
- Language mixing analysis (Cyrillic vs Latin)

### 4. Baselines — `notebooks/02_baselines.ipynb`

Trains and evaluates three TF-IDF classical baselines. Run all cells. Produces:
- Multinomial Naive Bayes: TF-IDF (50K features, bigrams, sublinear_tf) + MultinomialNB
- Logistic Regression: TF-IDF + LogReg (C=1.0, lbfgs)
- Linear SVM: TF-IDF + LinearSVC with CalibratedClassifierCV
- Confusion matrices, ROC curves, results comparison table
- Saves `results/baseline_metrics.json`

### 5. mDeBERTa-v3 — `notebooks/03_mdeberta.ipynb`

Fine-tunes `microsoft/mdeberta-v3-base`. **Run smoke test first (1000 samples, 1 epoch).**

Hyperparameters (matching Yeshpanov & Varol 2024):
- `learning_rate = 1e-5`
- `weight_decay = 1e-3`
- `batch_size = 16`
- `max_length = 128`
- `num_epochs = 3`
- `warmup_steps = 800`

Produces: training loss curves, confusion matrix, ROC curve, test metrics.
Saves: `results/transformer_metrics.json`, `results/test_predictions.csv`, model to `models/mdeberta-best/`.

**Hardware notes:**
- GPU recommended (A100, V100, RTX 3090/4090).
- Full 180K training: ~3-6 hours on GPU.
- Apple M1/M2 (MPS): may work but slow; reduce batch_size to 8 if OOM.
- CPU: not practical for full training.

### 6. Error Analysis — `notebooks/04_error_analysis.ipynb` (planned)

Will analyze misclassified examples.

## Key Technical Details

- **Random seed:** 42 everywhere (numpy, sklearn, torch, transformers)
- **Text input:** Uses `text_cleaned` column from KazSAnDRA (already preprocessed by authors)
- **TF-IDF config:** max_features=50000, ngram_range=(1,2), sublinear_tf=True
- **Transformer config:** lr=1e-5, weight_decay=1e-3, warmup=800, batch_size=16, max_length=128, epochs=3, best model by val F1 macro
- **Primary metric:** Macro F1 (matches the original paper's methodology)
- **SOTA target:** F1 = 0.81 (Yeshpanov & Varol 2024, XLM-R and RemBERT)
- **Keras/TF conflict fix:** `src/transformers_train.py` sets `USE_TF=0`

## Module Reference

### `src/data_loader.py`
- `load_kazsandra()` — load from HuggingFace, return dict of DataFrames (train/val/test)

### `src/preprocessing.py`
- `normalize_text(text)` — minimal normalization (lowercase, whitespace)

### `src/baselines.py`
- `build_mnb_pipeline(max_features, ngram_range)` — TF-IDF + MultinomialNB
- `build_logreg_pipeline(max_features, ngram_range)` — TF-IDF + LogReg
- `build_svm_pipeline(max_features, ngram_range)` — TF-IDF + LinearSVC (calibrated)

### `src/evaluation.py`
- `compute_metrics(y_true, y_pred, y_prob)` — returns dict with accuracy, precision, recall, F1, ROC-AUC
- `print_report(y_true, y_pred)` — prints sklearn classification report
- `plot_confusion_matrix(y_true, y_pred, title, ax)` — heatmap
- `plot_roc_curve(y_true, y_prob, label, ax)` — ROC curve
- `save_metrics(metrics_dict, filepath)` — save to JSON

### `src/transformers_train.py`
- `SentimentDataset(texts, labels, tokenizer, max_length)` — PyTorch Dataset
- `compute_metrics_hf(eval_pred)` — HuggingFace Trainer compatible
- `build_model_and_tokenizer(model_name)` — load pretrained model + tokenizer
- `get_training_args(output_dir, ...)` — TrainingArguments (defaults match KazSAnDRA paper)
- `get_predictions(trainer, dataset)` — returns (preds, probs, labels)

## Paper

Research paper uses the Springer Nature LaTeX template (`sn-jnl.cls`). Target: 10-14 pages.

Contributions:
1. First evaluation of classical ML baselines (MNB, LogReg, SVM) on KazSAnDRA
2. First evaluation of mDeBERTa-v3 on KazSAnDRA
3. Performance-vs-efficiency tradeoff analysis

## Execution Order Summary

```
1. pip install -r requirements.txt
2. python3 src/data_loader.py          # verify dataset downloads
3. Run notebooks/01_eda_kazsandra.ipynb
4. Run notebooks/02_baselines.ipynb
5. Run notebooks/03_mdeberta.ipynb     # smoke test first!
6. Run notebooks/04_error_analysis.ipynb
```
