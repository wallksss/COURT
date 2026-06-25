# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
# Aluno: Anderson Cristiano Sassaki Gonçalves e Lorenzo Grippo Chiachio
# RA: 821675 e 823917
# ################################################################
#!/usr/bin/env python3
"""Train Legal-BERTimbau Large with validation, then refit the final model.

This script extracts the notebook's best-effort Transformer path into a plain
Python entry point. It loads or rebuilds the preprocessed data, first trains
``rufimelo/Legal-BERTimbau-sts-large-ma-v3`` with a stratified validation split
to select the best checkpoint/epoch, then retrains the final model on all
available labeled rows without validation and writes Kaggle-style submissions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experimentos import (  # noqa: E402
    ExperimentConfig,
    auto_runtime_profile,
    combine_with_duplicate_prior,
    duplicate_prior_probabilities,
    estimate_transition_matrix,
    fine_tune_transformer_classifier,
    load_transformer_trainer,
    predict_transformer_probabilities,
    transductive_viterbi_submission,
)
from scripts.notebook_support import cached_preprocessed_data, resolve_project_dir  # noqa: E402
from scripts.preprocessamento import (  # noqa: E402
    TextPreprocessConfig,
    clean_body,
    fix_ocr_artifacts,
    normalize_legal_references,
    normalize_whitespace,
)


DEFAULT_MODEL_NAME = "rufimelo/Legal-BERTimbau-sts-large-ma-v3"
DEFAULT_VALID_LABELS = (0, 1, 2, 3, 4)
TRANSFORMER_TEXT_COL = "Body_transformer"


@dataclass(frozen=True)
class FinalTrainingConfig:
    """Resolved settings for the final no-validation training run."""

    project_dir: Path
    data_dir: Path
    output_dir: Path
    model_dir: Path
    model_name: str
    validation_model_dir: Path
    final_model_dir: Path
    raw_submission_path: Path
    final_submission_path: Path
    metadata_path: Path
    validation_report_path: Path
    pseudo_label_path: Path
    pseudo_label_report_path: Path
    pseudo_training_path: Path
    grid_report_path: Path
    grid_model_dir: Path
    epochs: float
    validation_size: float
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    weight_decay: float
    warmup_ratio: float
    seed: int
    device: str
    force_preprocess: bool
    force_retrain: bool
    force_predict: bool
    force_pseudo_label: bool
    force_grid_search: bool
    use_class_weights: bool
    gradient_checkpointing: bool
    use_viterbi: bool
    viterbi_lambda: float
    use_pseudo_labels: bool
    pseudo_label_confidence_threshold: float
    pseudo_label_margin_threshold: float
    pseudo_label_min_words: int
    run_grid_search: bool
    grid_learning_rates: tuple[float, ...]
    grid_epochs: tuple[float, ...]
    grid_warmup_ratios: tuple[float, ...]
    grid_weight_decays: tuple[float, ...]
    use_duplicate_prior: bool
    duplicate_lambda: float
    optimize_postprocessing: bool
    viterbi_lambda_grid: tuple[float, ...]
    duplicate_lambda_grid: tuple[float, ...]

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def build_preprocess_config() -> TextPreprocessConfig:
    """Return a cache config that preserves BERTimbau's cased Portuguese signals."""

    return TextPreprocessConfig(
        lowercase=False,
        strip_accents=False,
        normalize_legal_refs=True,
        remove_urls_emails=True,
        remove_stopwords=False,
        min_token_len=2,
        keep_digits=True,
    )


def build_experiment_config(output_dir: Path, seed: int) -> ExperimentConfig:
    """Return the modeling contract shared with the notebook helpers."""

    return ExperimentConfig(
        text_col=TRANSFORMER_TEXT_COL,
        target_col="Category",
        id_col="Id",
        random_state=int(seed),
        validation_size=0.2,
        cv_folds=5,
        scoring="f1_macro",
        output_dir=str(output_dir),
    )


def clean_transformer_text(text: object) -> str:
    """Clean text for BERT-like models without lowercasing, deaccenting or bag-of-words tokenization."""

    cleaned = clean_body(text)
    cleaned = fix_ocr_artifacts(cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " URL ", cleaned)
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " EMAIL ", cleaned)
    cleaned = normalize_legal_references(cleaned)
    return normalize_whitespace(cleaned)


def add_transformer_text_column(
    df: pd.DataFrame,
    source_col: str = "Body",
    fallback_col: str = "Body_clean",
    output_col: str = TRANSFORMER_TEXT_COL,
) -> pd.DataFrame:
    """Add the minimally cleaned text column used by the Transformer path."""

    result = df.copy()
    if source_col in result.columns:
        source = result[source_col]
    elif fallback_col in result.columns:
        source = result[fallback_col]
    else:
        raise KeyError(f"Neither '{source_col}' nor '{fallback_col}' was found.")
    result[output_col] = source.map(clean_transformer_text).fillna("").astype(str)
    return result


def load_preprocessed_frames(
    project_dir: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    config: TextPreprocessConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Load preprocessed train/test frames, rebuilding from raw CSVs when possible."""

    project_path = Path(project_dir).resolve()
    data_path = _resolve_path(project_path, data_dir)
    output_path = _resolve_path(project_path, output_dir)
    cache_dir = output_path / "cache"

    raw_csvs_available = (data_path / "train.csv").exists() and (data_path / "test.csv").exists()
    if force or raw_csvs_available:
        return cached_preprocessed_data(
            project_dir=project_path,
            data_dir=data_path,
            output_dir=output_path,
            config=config,
            force=force,
        )

    train_cache = cache_dir / "train_clean.csv"
    test_cache = cache_dir / "test_clean.csv"
    unlabeled_cache = cache_dir / "train_unlabeled_clean.csv"
    missing = [path for path in (train_cache, test_cache) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Raw CSV files were not found and the clean cache is incomplete. "
            f"Missing cache file(s): {missing_text}. Put train.csv/test.csv in "
            f"{data_path} or restore the preprocessing cache."
        )

    train_clean = pd.read_csv(train_cache)
    test_clean = pd.read_csv(test_cache)
    train_unlabeled_clean = pd.read_csv(unlabeled_cache) if unlabeled_cache.exists() else pd.DataFrame()
    return {
        "project_dir": project_path,
        "data_dir": data_path,
        "output_dir": output_path,
        "cache_dir": cache_dir,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "train_unlabeled_clean": train_unlabeled_clean,
        "cache_status": "loaded_cache_only",
    }


def select_training_frame(
    cache_dir: str | Path,
    train_clean: pd.DataFrame,
    target_col: str = "Category",
    valid_labels: Sequence[int] = DEFAULT_VALID_LABELS,
) -> tuple[pd.DataFrame, str]:
    """Pick the frame used for final training and drop unlabeled rows."""

    cache_path = Path(cache_dir)
    modeling_cache = cache_path / "train_modeling.csv"
    if modeling_cache.exists():
        training = pd.read_csv(modeling_cache)
        source = str(modeling_cache)
    else:
        training = train_clean.copy()
        source = "train_clean"

    if target_col not in training.columns:
        raise KeyError(f"Column '{target_col}' not found in the training frame.")
    if "Body_clean" not in training.columns:
        raise KeyError("Column 'Body_clean' not found in the training frame.")
    if TRANSFORMER_TEXT_COL not in training.columns:
        training = add_transformer_text_column(training)

    valid_label_set = set(int(label) for label in valid_labels)
    training = training.loc[training[target_col].isin(valid_label_set)].copy()
    if training.empty:
        raise ValueError("No rows with valid labels were found for final training.")
    training[target_col] = training[target_col].astype(int)
    return training.reset_index(drop=True), source


def align_probabilities_to_classes(
    probabilities: np.ndarray,
    label_values: Sequence[object],
    classes: Sequence[int],
) -> np.ndarray:
    """Align Transformer output columns to the desired integer class order."""

    probs = np.asarray(probabilities, dtype=float)
    class_values = [int(label) for label in classes]
    normalized_labels = [_coerce_label_value(label) for label in label_values]
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if probs.shape[1] != len(normalized_labels):
        raise ValueError("label_values length must match the number of probability columns.")

    class_to_col = {label: idx for idx, label in enumerate(class_values)}
    aligned = np.zeros((probs.shape[0], len(class_values)), dtype=float)
    for source_idx, label in enumerate(normalized_labels):
        target_idx = class_to_col.get(label)
        if target_idx is not None:
            aligned[:, target_idx] = probs[:, source_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    uniform = np.full_like(aligned, 1.0 / len(class_values))
    return np.divide(aligned, row_sums, out=uniform, where=row_sums > 0)


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Return compact validation metrics for labeled predictions."""

    from sklearn.metrics import accuracy_score, classification_report, f1_score

    return {
        "n_eval": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def select_pseudo_labeled_rows(
    unlabeled_df: pd.DataFrame,
    probabilities: np.ndarray,
    label_values: Sequence[object],
    classes: Sequence[int],
    text_col: str,
    target_col: str,
    confidence_threshold: float,
    margin_threshold: float,
    min_words: int,
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select high-confidence pseudo-labels from model probabilities."""

    if unlabeled_df.empty:
        return pd.DataFrame(), {
            "available_unlabeled": 0,
            "selected": 0,
            "coverage": 0.0,
            "confidence_threshold": float(confidence_threshold),
            "margin_threshold": float(margin_threshold),
            "min_words": int(min_words),
        }
    if text_col not in unlabeled_df.columns:
        raise KeyError(f"Column '{text_col}' not found in unlabeled_df.")

    aligned_probs = align_probabilities_to_classes(probabilities, label_values=label_values, classes=classes)
    class_values = np.asarray(classes, dtype=int)
    top_indices = np.argmax(aligned_probs, axis=1)
    top_probs = aligned_probs[np.arange(len(aligned_probs)), top_indices]
    sorted_probs = np.sort(aligned_probs, axis=1)
    second_probs = sorted_probs[:, -2] if aligned_probs.shape[1] > 1 else np.zeros(len(aligned_probs))
    margins = top_probs - second_probs
    word_counts = unlabeled_df[text_col].fillna("").astype(str).str.split().map(len).to_numpy()

    accepted = (
        (top_probs >= float(confidence_threshold))
        & (margins >= float(margin_threshold))
        & (word_counts >= int(min_words))
    )
    selected = unlabeled_df.loc[accepted].copy().reset_index(drop=True)
    if selected.empty:
        report = _pseudo_label_report(unlabeled_df, selected, accepted, word_counts, confidence_threshold, margin_threshold, min_words)
        return selected, report

    pseudo_categories = class_values[top_indices[accepted]].astype(int)
    selected["OriginalCategory"] = selected[target_col] if target_col in selected.columns else np.nan
    selected[target_col] = pseudo_categories
    selected["IsPseudoLabel"] = True
    selected["PseudoCategory"] = pseudo_categories
    selected["PseudoConfidence"] = top_probs[accepted].astype(float)
    selected["PseudoConfidenceMargin"] = margins[accepted].astype(float)
    selected["PseudoSource"] = source
    report = _pseudo_label_report(unlabeled_df, selected, accepted, word_counts, confidence_threshold, margin_threshold, min_words)
    report["selected_by_category"] = selected[target_col].value_counts().sort_index().astype(int).to_dict()
    return selected, report


def build_training_frame_with_pseudo_labels(
    train_df: pd.DataFrame,
    pseudo_df: pd.DataFrame | None,
    target_col: str = "Category",
) -> pd.DataFrame:
    """Append selected pseudo-labels while preserving traceability columns."""

    training = train_df.copy().reset_index(drop=True)
    training["IsPseudoLabel"] = False
    if "OriginalCategory" not in training.columns:
        training["OriginalCategory"] = np.nan
    if pseudo_df is None or pseudo_df.empty:
        training[target_col] = training[target_col].astype(int)
        return training

    pseudo = pseudo_df.copy().reset_index(drop=True)
    pseudo["IsPseudoLabel"] = True
    if "OriginalCategory" not in pseudo.columns:
        pseudo["OriginalCategory"] = np.nan
    missing_cols = [column for column in training.columns if column not in pseudo.columns]
    for column in missing_cols:
        pseudo[column] = np.nan
    extra_cols = [column for column in pseudo.columns if column not in training.columns]
    for column in extra_cols:
        training[column] = np.nan
    combined = pd.concat([training, pseudo[training.columns]], ignore_index=True, sort=False)
    combined[target_col] = combined[target_col].astype(int)
    return combined


def create_or_load_pseudo_labels(
    config: FinalTrainingConfig,
    experiment_config: ExperimentConfig,
    validation_trainer,
    unlabeled_df: pd.DataFrame,
    classes: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create or reuse high-confidence pseudo-labels from the validation model."""

    if not config.use_pseudo_labels:
        report = {"enabled": False, "selected": 0, "available_unlabeled": int(len(unlabeled_df))}
        return pd.DataFrame(), report
    if config.pseudo_label_path.exists() and config.pseudo_label_report_path.exists() and not config.force_pseudo_label:
        pseudo = pd.read_csv(config.pseudo_label_path)
        report = json.loads(config.pseudo_label_report_path.read_text(encoding="utf-8"))
        print(f"Pseudo-rotulos carregados de: {config.pseudo_label_path} ({len(pseudo)} linhas)")
        return pseudo, report
    if unlabeled_df.empty:
        report = {"enabled": True, "selected": 0, "available_unlabeled": 0, "reason": "no_unlabeled_rows"}
        save_pseudo_label_artifacts(config, pd.DataFrame(), report)
        return pd.DataFrame(), report

    print("Gerando pseudo-rotulos de alta confianca para linhas sem rotulo...")
    probabilities, label_values = predict_transformer_probabilities(
        validation_trainer,
        unlabeled_df,
        text_col=experiment_config.text_col,
        max_length=config.max_length,
    )
    pseudo, report = select_pseudo_labeled_rows(
        unlabeled_df,
        probabilities,
        label_values=label_values,
        classes=classes,
        text_col=experiment_config.text_col,
        target_col=experiment_config.target_col,
        confidence_threshold=config.pseudo_label_confidence_threshold,
        margin_threshold=config.pseudo_label_margin_threshold,
        min_words=config.pseudo_label_min_words,
        source=str(config.validation_model_dir),
    )
    report["enabled"] = True
    save_pseudo_label_artifacts(config, pseudo, report)
    print(f"Pseudo-rotulos selecionados: {report.get('selected', 0)} de {report.get('available_unlabeled', 0)}")
    return pseudo, report


def save_pseudo_label_artifacts(config: FinalTrainingConfig, pseudo: pd.DataFrame, report: dict[str, Any]) -> None:
    config.pseudo_label_path.parent.mkdir(parents=True, exist_ok=True)
    config.pseudo_label_report_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo.to_csv(config.pseudo_label_path, index=False)
    config.pseudo_label_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_float_grid(value: str | Sequence[float]) -> tuple[float, ...]:
    """Parse comma-separated float grids from the CLI."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(float(part) for part in parts)
    return tuple(float(item) for item in value)


def build_grid_candidates(config: FinalTrainingConfig) -> list[dict[str, float]]:
    """Create hyperparameter grid candidates."""

    candidates = []
    for learning_rate, epochs, warmup_ratio, weight_decay in product(
        config.grid_learning_rates or (config.learning_rate,),
        config.grid_epochs or (config.epochs,),
        config.grid_warmup_ratios or (config.warmup_ratio,),
        config.grid_weight_decays or (config.weight_decay,),
    ):
        candidates.append(
            {
                "learning_rate": float(learning_rate),
                "epochs": float(epochs),
                "warmup_ratio": float(warmup_ratio),
                "weight_decay": float(weight_decay),
            }
        )
    return candidates


def run_hyperparameter_grid_search(
    config: FinalTrainingConfig,
    experiment_config: ExperimentConfig,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run an optional validation grid and persist every trial."""

    if config.grid_report_path.exists() and not config.force_grid_search and not config.force_retrain:
        grid_summary = json.loads(config.grid_report_path.read_text(encoding="utf-8"))
        print(f"Grid search carregado de: {config.grid_report_path}")
        return grid_summary

    config.grid_model_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    best: dict[str, Any] | None = None
    for trial_idx, candidate in enumerate(build_grid_candidates(config), start=1):
        trial_dir = config.grid_model_dir / f"trial_{trial_idx:03d}"
        print(f"Grid trial {trial_idx}: {candidate}")
        trainer = fine_tune_transformer_classifier(
            train_df,
            model_name=config.model_name,
            config=experiment_config,
            num_train_epochs=candidate["epochs"],
            learning_rate=candidate["learning_rate"],
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_length,
            output_dir=str(trial_dir),
            weight_decay=candidate["weight_decay"],
            warmup_ratio=candidate["warmup_ratio"],
            gradient_checkpointing=config.gradient_checkpointing,
            use_class_weights=config.use_class_weights,
            device=config.device,
            validation_size=config.validation_size,
        )
        trainer.save_model(trial_dir)
        tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(trial_dir)
        summary = extract_validation_summary(trainer, validation_size=config.validation_size, default_epoch=candidate["epochs"])
        row = {
            "trial": trial_idx,
            "model_dir": str(trial_dir),
            **candidate,
            "best_metric": summary.get("best_metric"),
            "best_epoch": summary.get("best_epoch"),
            "best_model_checkpoint": summary.get("best_model_checkpoint"),
        }
        trials.append(row)
        if best is None or float(row.get("best_metric") or float("-inf")) > float(best.get("best_metric") or float("-inf")):
            best = row
        config.grid_report_path.parent.mkdir(parents=True, exist_ok=True)
        config.grid_report_path.write_text(
            json.dumps({"best": best, "trials": trials}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {"best": best, "trials": trials}


def apply_grid_search_result(config: FinalTrainingConfig, grid_summary: dict[str, Any]) -> FinalTrainingConfig:
    """Return a config updated with the best grid-search candidate."""

    best = grid_summary.get("best") or {}
    if not best:
        return config
    print("Melhores hiperparametros do grid:", best)
    return replace(
        config,
        learning_rate=float(best.get("learning_rate", config.learning_rate)),
        epochs=float(best.get("best_epoch") or best.get("epochs", config.epochs)),
        warmup_ratio=float(best.get("warmup_ratio", config.warmup_ratio)),
        weight_decay=float(best.get("weight_decay", config.weight_decay)),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Legal-BERTimbau Large, then retrain the final model on all labeled data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT, help="Project directory containing scripts/.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with train.csv/test.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for outputs and cache.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for saved models.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HuggingFace model id or local model path.")
    parser.add_argument(
        "--validation-model-dir-name",
        default="bert_large_validation",
        help="Saved validation-stage best model folder.",
    )
    parser.add_argument("--final-model-dir-name", default="bert_large_final_full_train", help="Saved final model folder.")
    parser.add_argument("--raw-submission-name", default="submission_bert_large_raw.csv", help="Raw argmax submission file.")
    parser.add_argument("--submission-name", default="submission_bert_large_final.csv", help="Final submission file.")
    parser.add_argument("--metadata-name", default="bert_large_final_metadata.json", help="Run metadata file.")
    parser.add_argument("--validation-report-name", default="bert_large_validation_metrics.json", help="Validation report file.")
    parser.add_argument("--pseudo-label-name", default="pseudo_labels_bert_large.csv", help="Pseudo-label cache file.")
    parser.add_argument(
        "--pseudo-label-report-name",
        default="pseudo_labels_bert_large_report.json",
        help="Pseudo-label report file.",
    )
    parser.add_argument(
        "--pseudo-training-name",
        default="train_modeling_bert_large_pseudo.csv",
        help="Final training frame with selected pseudo-labels.",
    )
    parser.add_argument("--grid-report-name", default="bert_large_grid_search.json", help="Grid-search report file.")
    parser.add_argument("--grid-model-dir-name", default="bert_large_grid_search", help="Grid-search model folder.")
    parser.add_argument("--epochs", type=float, default=5.0, help="Maximum number of validation-stage epochs.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Stratified validation split size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train/eval batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="LR warmup ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Training device.")
    parser.add_argument("--viterbi-lambda", type=float, default=0.75, help="Sequential post-processing strength.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild preprocessing cache from raw CSVs.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if the final model folder exists.")
    parser.add_argument("--force-predict", action="store_true", help="Rewrite submission files even if they exist.")
    parser.add_argument("--force-pseudo-label", action="store_true", help="Regenerate pseudo-labels even if cached.")
    parser.add_argument("--force-grid-search", action="store_true", help="Regenerate grid-search results even if cached.")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false", help="Disable class weights.")
    parser.add_argument("--no-pseudo-labels", dest="use_pseudo_labels", action="store_false", help="Disable pseudo-labeling.")
    parser.add_argument(
        "--pseudo-label-confidence",
        type=float,
        default=0.97,
        help="Minimum top-class probability for pseudo-labels.",
    )
    parser.add_argument(
        "--pseudo-label-margin",
        type=float,
        default=0.30,
        help="Minimum top1-top2 probability margin for pseudo-labels.",
    )
    parser.add_argument(
        "--pseudo-label-min-words",
        type=int,
        default=2,
        help="Minimum cleaned word count for pseudo-label candidates.",
    )
    parser.add_argument("--run-grid-search", action="store_true", help="Run a small validation grid before final training.")
    parser.add_argument("--grid-learning-rates", default="5e-6,1e-5,2e-5", help="Comma-separated LR grid.")
    parser.add_argument("--grid-epochs", default="3,4,5", help="Comma-separated epoch grid.")
    parser.add_argument("--grid-warmup-ratios", default="0.0,0.03,0.06", help="Comma-separated warmup-ratio grid.")
    parser.add_argument("--grid-weight-decays", default="0.01", help="Comma-separated weight-decay grid.")
    parser.add_argument("--no-duplicate-prior", dest="use_duplicate_prior", action="store_false", help="Disable exact-text duplicate priors.")
    parser.add_argument("--duplicate-lambda", type=float, default=1.0, help="Exact-text duplicate prior strength.")
    parser.add_argument(
        "--no-optimize-postprocessing",
        dest="optimize_postprocessing",
        action="store_false",
        help="Disable validation-only tuning of duplicate/Viterbi strengths.",
    )
    parser.add_argument(
        "--viterbi-lambda-grid",
        default="0.0,0.25,0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated Viterbi lambda grid for validation-only post-processing tuning.",
    )
    parser.add_argument(
        "--duplicate-lambda-grid",
        default="0.0,0.25,0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated duplicate-prior lambda grid for validation-only post-processing tuning.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument("--no-viterbi", dest="use_viterbi", action="store_false", help="Use raw Transformer argmax output.")
    parser.set_defaults(
        use_class_weights=True,
        gradient_checkpointing=True,
        use_viterbi=True,
        use_pseudo_labels=True,
        use_duplicate_prior=True,
        optimize_postprocessing=True,
    )
    return parser.parse_args(argv)


def resolve_training_config(args: argparse.Namespace) -> FinalTrainingConfig:
    """Resolve relative paths and automatic runtime settings."""

    requested_project_dir = Path(args.project_dir).resolve()
    project_dir = resolve_project_dir(requested_project_dir)
    output_dir = _resolve_path(project_dir, args.output_dir)
    model_dir = _resolve_path(project_dir, args.model_dir)
    data_dir = _resolve_path(project_dir, args.data_dir)

    runtime_profile = auto_runtime_profile(prefer_transformer=True)
    detected_device = str(runtime_profile.get("device", "cpu"))
    device = detected_device if args.device == "auto" else args.device
    batch_size = args.batch_size if args.batch_size is not None else _default_batch_size(device)
    grad_accum = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else _default_gradient_accumulation(device)
    )

    return FinalTrainingConfig(
        project_dir=project_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        model_name=args.model_name,
        validation_model_dir=model_dir / args.validation_model_dir_name,
        final_model_dir=model_dir / args.final_model_dir_name,
        raw_submission_path=output_dir / args.raw_submission_name,
        final_submission_path=output_dir / args.submission_name,
        metadata_path=output_dir / args.metadata_name,
        validation_report_path=output_dir / args.validation_report_name,
        pseudo_label_path=output_dir / "cache" / args.pseudo_label_name,
        pseudo_label_report_path=output_dir / args.pseudo_label_report_name,
        pseudo_training_path=output_dir / "cache" / args.pseudo_training_name,
        grid_report_path=output_dir / args.grid_report_name,
        grid_model_dir=model_dir / args.grid_model_dir_name,
        epochs=float(args.epochs),
        validation_size=float(args.validation_size),
        learning_rate=float(args.learning_rate),
        batch_size=int(batch_size),
        gradient_accumulation_steps=int(grad_accum),
        max_length=int(args.max_length),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        seed=int(args.seed),
        device=str(device),
        force_preprocess=bool(args.force_preprocess),
        force_retrain=bool(args.force_retrain),
        force_predict=bool(args.force_predict),
        force_pseudo_label=bool(args.force_pseudo_label),
        force_grid_search=bool(args.force_grid_search),
        use_class_weights=bool(args.use_class_weights),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        use_viterbi=bool(args.use_viterbi),
        viterbi_lambda=float(args.viterbi_lambda),
        use_pseudo_labels=bool(args.use_pseudo_labels),
        pseudo_label_confidence_threshold=float(args.pseudo_label_confidence),
        pseudo_label_margin_threshold=float(args.pseudo_label_margin),
        pseudo_label_min_words=int(args.pseudo_label_min_words),
        run_grid_search=bool(args.run_grid_search),
        grid_learning_rates=parse_float_grid(args.grid_learning_rates),
        grid_epochs=parse_float_grid(args.grid_epochs),
        grid_warmup_ratios=parse_float_grid(args.grid_warmup_ratios),
        grid_weight_decays=parse_float_grid(args.grid_weight_decays),
        use_duplicate_prior=bool(args.use_duplicate_prior),
        duplicate_lambda=float(args.duplicate_lambda),
        optimize_postprocessing=bool(args.optimize_postprocessing),
        viterbi_lambda_grid=parse_float_grid(args.viterbi_lambda_grid),
        duplicate_lambda_grid=parse_float_grid(args.duplicate_lambda_grid),
    )


def validation_holdout_split(
    train_df: pd.DataFrame,
    experiment_config: ExperimentConfig,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the Trainer's stratified holdout split for post-processing calibration."""

    from sklearn.model_selection import train_test_split

    if validation_size <= 0:
        raise ValueError("validation_size must be positive.")
    labels = train_df[experiment_config.target_col].astype(int).reset_index(drop=True)
    indices = np.arange(len(train_df))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=float(validation_size),
        random_state=experiment_config.random_state,
        stratify=labels,
    )
    return (
        train_df.iloc[train_idx].reset_index(drop=True),
        train_df.iloc[valid_idx].reset_index(drop=True),
    )


def optimize_postprocessing_on_validation(
    config: FinalTrainingConfig,
    experiment_config: ExperimentConfig,
    validation_trainer,
    train_df: pd.DataFrame,
    classes: Sequence[int],
) -> dict[str, Any]:
    """Tune text-only duplicate and sequence post-processing on the validation holdout."""

    if not config.optimize_postprocessing:
        return {"enabled": False, "reason": "disabled"}
    if config.validation_size <= 0:
        return {"enabled": False, "reason": "no_validation_split"}
    if not config.use_duplicate_prior and not config.use_viterbi:
        return {"enabled": False, "reason": "all_postprocessing_disabled"}

    from sklearn.metrics import f1_score

    try:
        train_part, valid_part = validation_holdout_split(train_df, experiment_config, config.validation_size)
    except ValueError as exc:
        return {"enabled": False, "reason": f"validation_split_unavailable: {exc}"}
    probabilities, label_values = predict_transformer_probabilities(
        validation_trainer,
        valid_part,
        text_col=experiment_config.text_col,
        max_length=config.max_length,
    )
    aligned_probs = align_probabilities_to_classes(probabilities, label_values=label_values, classes=classes)
    class_values = np.asarray(classes, dtype=int)
    y_true = valid_part[experiment_config.target_col].astype(int).to_numpy()

    duplicate_probs = None
    duplicate_grid = (0.0,)
    if config.use_duplicate_prior:
        duplicate_probs = duplicate_prior_probabilities(
            train_part,
            target_texts=valid_part[experiment_config.text_col],
            classes=classes,
            text_col=experiment_config.text_col,
            target_col=experiment_config.target_col,
            alpha=1.0,
        )
        duplicate_grid = config.duplicate_lambda_grid or (config.duplicate_lambda,)

    viterbi_grid = config.viterbi_lambda_grid if config.use_viterbi else (0.0,)
    viterbi_grid = viterbi_grid or (config.viterbi_lambda,)
    transition = estimate_transition_matrix(
        train_part[experiment_config.target_col].astype(int),
        classes=classes,
        ids=train_part[experiment_config.id_col],
        alpha=1.0,
    )

    base_predictions = class_values[np.argmax(aligned_probs, axis=1)]
    base_f1 = f1_score(y_true, base_predictions, average="macro", zero_division=0)
    candidates = []
    best: dict[str, Any] | None = None
    for duplicate_lambda in duplicate_grid:
        candidate_probs = (
            combine_with_duplicate_prior(aligned_probs, duplicate_probs, lambda_dup=float(duplicate_lambda))
            if duplicate_probs is not None
            else aligned_probs
        )
        for viterbi_lambda in viterbi_grid:
            if config.use_viterbi:
                decoded = transductive_viterbi_submission(
                    train_part,
                    valid_part,
                    candidate_probs,
                    classes=classes,
                    config=experiment_config,
                    transition_probs=transition,
                    lambda_transition=float(viterbi_lambda),
                    output_path=None,
                )
                predictions = decoded["Category"].astype(int).to_numpy()
            else:
                predictions = class_values[np.argmax(candidate_probs, axis=1)]

            score = f1_score(y_true, predictions, average="macro", zero_division=0)
            candidate = {
                "duplicate_lambda": float(duplicate_lambda),
                "viterbi_lambda": float(viterbi_lambda),
                "f1_macro": float(score),
            }
            candidates.append(candidate)
            if best is None or candidate["f1_macro"] > best["f1_macro"]:
                best = candidate

    best = best or {"duplicate_lambda": config.duplicate_lambda, "viterbi_lambda": config.viterbi_lambda, "f1_macro": base_f1}
    return {
        "enabled": True,
        "validation_rows": int(len(valid_part)),
        "known_train_rows": int(len(train_part)),
        "base_argmax_f1_macro": float(base_f1),
        "duplicate_lambda": float(best["duplicate_lambda"]),
        "viterbi_lambda": float(best["viterbi_lambda"]),
        "best_f1_macro": float(best["f1_macro"]),
        "candidates": candidates,
    }


def apply_postprocessing_result(config: FinalTrainingConfig, validation_summary: dict[str, Any]) -> FinalTrainingConfig:
    """Apply stored validation-selected post-processing strengths to the final config."""

    postprocessing = validation_summary.get("postprocessing") or {}
    if not postprocessing.get("enabled"):
        return config
    return replace(
        config,
        duplicate_lambda=float(postprocessing.get("duplicate_lambda", config.duplicate_lambda)),
        viterbi_lambda=float(postprocessing.get("viterbi_lambda", config.viterbi_lambda)),
    )


def run_final_training(config: FinalTrainingConfig) -> dict[str, Any]:
    """Run preprocessing, final no-validation training, prediction and metadata."""

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-cache")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "cache").mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)

    preprocess_config = build_preprocess_config()
    experiment_config = build_experiment_config(config.output_dir, config.seed)
    cache_bundle = load_preprocessed_frames(
        project_dir=config.project_dir,
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        config=preprocess_config,
        force=config.force_preprocess,
    )
    train_clean = add_transformer_text_column(cache_bundle["train_clean"])
    test_clean = add_transformer_text_column(cache_bundle["test_clean"])
    train_df, training_source = select_training_frame(
        cache_dir=cache_bundle["cache_dir"],
        train_clean=train_clean,
        target_col=experiment_config.target_col,
    )
    train_unlabeled_clean = cache_bundle.get("train_unlabeled_clean", pd.DataFrame()).copy()
    train_unlabeled_df = (
        add_transformer_text_column(train_unlabeled_clean)
        if not train_unlabeled_clean.empty
        else train_unlabeled_clean
    )
    classes = sorted(train_df[experiment_config.target_col].astype(int).unique().tolist())

    grid_summary: dict[str, Any] | None = None
    if config.run_grid_search:
        grid_summary = run_hyperparameter_grid_search(config, experiment_config, train_df)
        config = apply_grid_search_result(config, grid_summary)

    print_run_header(config, cache_bundle, train_df, test_clean, training_source, classes)
    model_existed_before = config.final_model_dir.exists()
    trainer, validation_summary = load_or_train_final_transformer(
        config,
        experiment_config,
        train_df,
        train_unlabeled_df=train_unlabeled_df,
        classes=classes,
    )
    if grid_summary is not None:
        validation_summary["grid_search"] = grid_summary
    config = apply_postprocessing_result(config, validation_summary)
    trained_new_model = config.force_retrain or not model_existed_before
    submissions = write_submissions(
        config,
        experiment_config,
        trainer,
        train_df,
        test_clean,
        classes,
        force_write=trained_new_model,
    )
    metadata = write_metadata(
        config=config,
        preprocess_config=preprocess_config,
        experiment_config=experiment_config,
        cache_bundle=cache_bundle,
        training_source=training_source,
        train_rows=len(train_df),
        test_rows=len(test_clean),
        classes=classes,
        submissions=submissions,
        validation_summary=validation_summary,
    )
    return {
        "config": config,
        "metadata": metadata,
        "submissions": submissions,
        "train_rows": len(train_df),
        "test_rows": len(test_clean),
    }


def load_or_train_final_transformer(
    config: FinalTrainingConfig,
    experiment_config: ExperimentConfig,
    train_df: pd.DataFrame,
    train_unlabeled_df: pd.DataFrame | None = None,
    classes: Sequence[int] | None = None,
):
    """Load an existing final Trainer or validate and refit the large BERT."""

    if config.final_model_dir.exists() and not config.force_retrain:
        print(f"Carregando modelo final existente: {config.final_model_dir}")
        return load_transformer_trainer(config.final_model_dir, device=config.device), load_validation_summary(config)

    class_values = list(classes) if classes is not None else sorted(train_df[experiment_config.target_col].astype(int).unique())
    validation_trainer, validation_summary = load_or_train_validation_transformer(config, experiment_config, train_df)
    if config.force_retrain or "postprocessing" not in validation_summary:
        validation_summary["postprocessing"] = optimize_postprocessing_on_validation(
            config,
            experiment_config,
            validation_trainer,
            train_df,
            class_values,
        )
    pseudo_labels, pseudo_report = create_or_load_pseudo_labels(
        config,
        experiment_config,
        validation_trainer,
        train_unlabeled_df if train_unlabeled_df is not None else pd.DataFrame(),
        class_values,
    )
    final_train_df = build_training_frame_with_pseudo_labels(
        train_df,
        pseudo_labels,
        target_col=experiment_config.target_col,
    )
    config.pseudo_training_path.parent.mkdir(parents=True, exist_ok=True)
    final_train_df.to_csv(config.pseudo_training_path, index=False)

    final_epochs = float(validation_summary.get("best_epoch") or config.epochs)
    if final_epochs <= 0:
        final_epochs = config.epochs

    print("Treinando modelo final com todos os dados ampliados, sem validacao...")
    trainer = fine_tune_transformer_classifier(
        final_train_df,
        model_name=config.model_name,
        config=experiment_config,
        num_train_epochs=final_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        output_dir=str(config.final_model_dir),
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_checkpointing=config.gradient_checkpointing,
        use_class_weights=config.use_class_weights,
        device=config.device,
        validation_size=0.0,
    )
    trainer.save_model(config.final_model_dir)
    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(config.final_model_dir)
    validation_summary["final_epochs"] = float(final_epochs)
    validation_summary["final_train_rows"] = int(len(final_train_df))
    validation_summary["pseudo_label_report"] = pseudo_report
    validation_summary["pseudo_training_path"] = str(config.pseudo_training_path)
    save_validation_summary(config, validation_summary)
    return trainer, validation_summary


def load_or_train_validation_transformer(
    config: FinalTrainingConfig,
    experiment_config: ExperimentConfig,
    train_df: pd.DataFrame,
) -> tuple[Any, dict[str, Any]]:
    """Load or train the validation-stage model and return its summary."""

    if config.validation_model_dir.exists() and not config.force_retrain:
        cached_summary = load_validation_summary(config)
        checkpoint_summary = load_validation_checkpoint_summary(config)
        summary = cached_summary if cached_summary.get("uses_validation") is not None else checkpoint_summary
        checkpoint = summary.get("best_model_checkpoint") or summary.get("model_checkpoint") or str(config.validation_model_dir)
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists() and (checkpoint_path / "config.json").exists():
            print(f"Carregando modelo de validacao existente: {checkpoint_path}")
            save_validation_summary(config, summary)
            return load_transformer_trainer(checkpoint_path, device=config.device), summary

    if config.validation_size <= 0:
        print("Treinando modelo auxiliar sem validacao para etapas posteriores...")
        config.validation_model_dir.mkdir(parents=True, exist_ok=True)
        validation_trainer = fine_tune_transformer_classifier(
            train_df,
            model_name=config.model_name,
            config=experiment_config,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_length,
            output_dir=str(config.validation_model_dir),
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            gradient_checkpointing=config.gradient_checkpointing,
            use_class_weights=config.use_class_weights,
            device=config.device,
            validation_size=0.0,
        )
        validation_trainer.save_model(config.validation_model_dir)
        tokenizer = getattr(validation_trainer, "processing_class", None) or getattr(validation_trainer, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(config.validation_model_dir)
        summary = {
            "uses_validation": False,
            "validation_size": 0.0,
            "best_metric": None,
            "best_epoch": float(config.epochs),
            "best_model_checkpoint": None,
            "history": [],
        }
        save_validation_summary(config, summary)
        return validation_trainer, summary

    print("Treinando com validacao para selecionar o melhor checkpoint...")
    config.validation_model_dir.mkdir(parents=True, exist_ok=True)
    validation_trainer = fine_tune_transformer_classifier(
        train_df,
        model_name=config.model_name,
        config=experiment_config,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        output_dir=str(config.validation_model_dir),
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_checkpointing=config.gradient_checkpointing,
        use_class_weights=config.use_class_weights,
        device=config.device,
        validation_size=config.validation_size,
    )
    validation_trainer.save_model(config.validation_model_dir)
    tokenizer = getattr(validation_trainer, "processing_class", None) or getattr(validation_trainer, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(config.validation_model_dir)

    summary = extract_validation_summary(
        validation_trainer,
        validation_size=config.validation_size,
        default_epoch=config.epochs,
    )
    save_validation_summary(config, summary)
    print(
        "Melhor validacao:",
        f"f1_macro={summary.get('best_metric')}",
        f"epoch={summary.get('best_epoch')}",
    )
    return validation_trainer, summary


def run_validation_training(
    config: FinalTrainingConfig,
    experiment_config: ExperimentConfig,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train/load validation model and return only its summary."""

    _, summary = load_or_train_validation_transformer(config, experiment_config, train_df)
    return summary


def extract_validation_summary(
    trainer,
    validation_size: float,
    default_epoch: float,
) -> dict[str, Any]:
    """Extract the best validation metric/epoch from a HuggingFace Trainer."""

    state = getattr(trainer, "state", None)
    raw_history = list(getattr(state, "log_history", []) or [])
    history = [_json_safe_dict(row) for row in raw_history if isinstance(row, dict)]
    eval_rows = [row for row in history if row.get("eval_f1_macro") is not None]
    best_row = max(eval_rows, key=lambda row: float(row.get("eval_f1_macro", float("-inf"))), default={})
    state_best_metric = getattr(state, "best_metric", None)
    best_metric = state_best_metric if state_best_metric is not None else best_row.get("eval_f1_macro")
    best_epoch = best_row.get("epoch") or getattr(state, "epoch", None) or default_epoch
    return {
        "uses_validation": True,
        "validation_size": float(validation_size),
        "metric_for_best_model": "f1_macro",
        "best_metric": _json_safe_number(best_metric),
        "best_epoch": float(best_epoch),
        "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
        "history": history,
    }


def save_validation_summary(config: FinalTrainingConfig, summary: dict[str, Any]) -> None:
    """Persist validation-stage metrics for reproducibility."""

    config.validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.validation_report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def load_validation_summary(config: FinalTrainingConfig) -> dict[str, Any]:
    """Load the validation report when reusing an existing final model."""

    if config.validation_report_path.exists():
        return json.loads(config.validation_report_path.read_text(encoding="utf-8"))
    return {
        "uses_validation": None,
        "validation_size": float(config.validation_size),
        "best_metric": None,
        "best_epoch": None,
        "best_model_checkpoint": None,
        "history": [],
        "note": "validation report not found for the reused final model",
    }


def load_validation_checkpoint_summary(config: FinalTrainingConfig) -> dict[str, Any]:
    """Recover validation metrics from HuggingFace trainer_state.json checkpoints."""

    state_paths = sorted(config.validation_model_dir.glob("**/trainer_state.json"))
    best_state: dict[str, Any] | None = None
    best_metric = float("-inf")
    for state_path in state_paths:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metric = state.get("best_metric")
        if metric is None:
            eval_rows = [row for row in state.get("log_history", []) if row.get("eval_f1_macro") is not None]
            metric = max((row["eval_f1_macro"] for row in eval_rows), default=None)
        if metric is not None and float(metric) > best_metric:
            best_metric = float(metric)
            best_state = state
            best_state["_state_path"] = str(state_path)

    if best_state is None:
        return load_validation_summary(config)

    history = [_json_safe_dict(row) for row in best_state.get("log_history", []) if isinstance(row, dict)]
    eval_rows = [row for row in history if row.get("eval_f1_macro") is not None]
    best_row = max(eval_rows, key=lambda row: float(row.get("eval_f1_macro", float("-inf"))), default={})
    checkpoint = best_state.get("best_model_checkpoint")
    if checkpoint is None:
        checkpoint = str(Path(best_state["_state_path"]).parent)
    return {
        "uses_validation": True,
        "validation_size": float(config.validation_size),
        "metric_for_best_model": "f1_macro",
        "best_metric": _json_safe_number(best_state.get("best_metric", best_row.get("eval_f1_macro"))),
        "best_epoch": float(best_row.get("epoch") or best_state.get("epoch") or config.epochs),
        "best_model_checkpoint": checkpoint,
        "model_checkpoint": checkpoint,
        "history": history,
        "recovered_from": best_state.get("_state_path"),
    }


def write_submissions(
    config: FinalTrainingConfig,
    experiment_config: ExperimentConfig,
    trainer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: Sequence[int],
    force_write: bool = False,
) -> dict[str, str]:
    """Write raw Transformer and final submission files."""

    should_reuse = (
        config.raw_submission_path.exists()
        and config.final_submission_path.exists()
        and not force_write
        and not config.force_predict
        and not config.force_retrain
    )
    if should_reuse:
        print(f"Submissoes existentes mantidas: {config.final_submission_path}")
        return {
            "raw_transformer": str(config.raw_submission_path),
            "final": str(config.final_submission_path),
        }

    print("Gerando predicoes do conjunto de teste...")
    probabilities, label_values = predict_transformer_probabilities(
        trainer,
        test_df,
        text_col=experiment_config.text_col,
        max_length=config.max_length,
    )
    aligned_probs = align_probabilities_to_classes(probabilities, label_values=label_values, classes=classes)
    class_values = np.asarray(classes, dtype=int)
    raw_labels = class_values[np.argmax(aligned_probs, axis=1)]
    raw_submission = pd.DataFrame(
        {
            experiment_config.id_col: test_df[experiment_config.id_col].values,
            "Category": raw_labels.astype(int),
        }
    )
    raw_submission.to_csv(config.raw_submission_path, index=False)

    final_probs = aligned_probs
    if config.use_duplicate_prior:
        duplicate_probs = duplicate_prior_probabilities(
            train_df,
            target_texts=test_df[experiment_config.text_col],
            classes=classes,
            text_col=experiment_config.text_col,
            target_col=experiment_config.target_col,
            alpha=1.0,
        )
        final_probs = combine_with_duplicate_prior(
            aligned_probs,
            duplicate_probs,
            lambda_dup=config.duplicate_lambda,
        )

    if config.use_viterbi:
        transition = estimate_transition_matrix(
            train_df[experiment_config.target_col].astype(int),
            classes=classes,
            ids=train_df[experiment_config.id_col],
            alpha=1.0,
        )
        final_submission = transductive_viterbi_submission(
            train_df,
            test_df,
            final_probs,
            classes=classes,
            config=experiment_config,
            transition_probs=transition,
            lambda_transition=config.viterbi_lambda,
            output_path=config.final_submission_path,
        )
    else:
        final_labels = class_values[np.argmax(final_probs, axis=1)]
        final_submission = pd.DataFrame(
            {
                experiment_config.id_col: test_df[experiment_config.id_col].values,
                "Category": final_labels.astype(int),
            }
        )
        final_submission.to_csv(config.final_submission_path, index=False)

    print(f"Submissao raw salva em: {config.raw_submission_path}")
    print(f"Submissao final salva em: {config.final_submission_path}")
    print(f"Linhas na submissao final: {len(final_submission)}")
    return {
        "raw_transformer": str(config.raw_submission_path),
        "final": str(config.final_submission_path),
    }


def write_metadata(
    config: FinalTrainingConfig,
    preprocess_config: TextPreprocessConfig,
    experiment_config: ExperimentConfig,
    cache_bundle: dict[str, Any],
    training_source: str,
    train_rows: int,
    test_rows: int,
    classes: Sequence[int],
    submissions: dict[str, str],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Persist a compact reproducibility record for the final run."""

    metadata = {
        "model_name": config.model_name,
        "final_training_uses_validation": False,
        "selection_training_uses_validation": validation_summary.get("uses_validation"),
        "project_dir": str(config.project_dir),
        "data_dir": str(config.data_dir),
        "cache_status": cache_bundle.get("cache_status"),
        "training_source": training_source,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "classes": [int(label) for label in classes],
        "preprocess_config": asdict(preprocess_config),
        "experiment_config": asdict(experiment_config),
        "training_config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "max_length": config.max_length,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "seed": config.seed,
            "device": config.device,
            "validation_size": config.validation_size,
            "final_epochs": validation_summary.get("final_epochs", validation_summary.get("best_epoch")),
            "use_class_weights": config.use_class_weights,
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_viterbi": config.use_viterbi,
            "viterbi_lambda": config.viterbi_lambda,
            "use_pseudo_labels": config.use_pseudo_labels,
            "pseudo_label_confidence_threshold": config.pseudo_label_confidence_threshold,
            "pseudo_label_margin_threshold": config.pseudo_label_margin_threshold,
            "pseudo_label_min_words": config.pseudo_label_min_words,
            "run_grid_search": config.run_grid_search,
            "use_duplicate_prior": config.use_duplicate_prior,
            "duplicate_lambda": config.duplicate_lambda,
            "optimize_postprocessing": config.optimize_postprocessing,
            "viterbi_lambda_grid": list(config.viterbi_lambda_grid),
            "duplicate_lambda_grid": list(config.duplicate_lambda_grid),
        },
        "validation": validation_summary,
        "artifacts": {
            "validation_model_dir": str(config.validation_model_dir),
            "validation_report": str(config.validation_report_path),
            "pseudo_labels": str(config.pseudo_label_path),
            "pseudo_label_report": str(config.pseudo_label_report_path),
            "pseudo_training": str(config.pseudo_training_path),
            "grid_report": str(config.grid_report_path),
            "model_dir": str(config.final_model_dir),
            **submissions,
        },
    }
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadados salvos em: {config.metadata_path}")
    return metadata


def print_run_header(
    config: FinalTrainingConfig,
    cache_bundle: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    training_source: str,
    classes: Sequence[int],
) -> None:
    print("Projeto:", config.project_dir)
    print("Cache:", cache_bundle["cache_dir"])
    print("Status do pre-processamento:", cache_bundle.get("cache_status"))
    print("Modelo:", config.model_name)
    print("Dispositivo:", config.device)
    print("Fonte de treino:", training_source)
    print("Linhas de treino final:", len(train_df))
    print("Linhas de teste:", len(test_df))
    print("Classes:", list(classes))
    print("Epocas:", config.epochs)
    print("Batch efetivo:", config.effective_batch_size)
    print("Validacao inicial:", config.validation_size)
    print("Treino final: sem validacao (validation_size=0.0)")
    print("Pseudo-labels:", "ativados" if config.use_pseudo_labels else "desativados")
    if config.use_pseudo_labels:
        print("Limiares pseudo-label:", config.pseudo_label_confidence_threshold, config.pseudo_label_margin_threshold)
    print("Prior de duplicata textual:", "ativado" if config.use_duplicate_prior else "desativado")
    if config.use_duplicate_prior:
        print("Lambda duplicata:", config.duplicate_lambda)
    print("Viterbi:", "ativado" if config.use_viterbi else "desativado", "lambda=", config.viterbi_lambda)
    print("Otimizacao de pos-processamento:", "ativada" if config.optimize_postprocessing else "desativada")
    print("Grid search:", "ativado" if config.run_grid_search else "desativado")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch when available."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(project_dir: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_dir / path


def _coerce_label_value(label: object) -> int:
    try:
        return int(label)
    except (TypeError, ValueError):
        value = str(label)
        if value.startswith("LABEL_"):
            return int(value.replace("LABEL_", "", 1))
        raise


def _pseudo_label_report(
    unlabeled_df: pd.DataFrame,
    selected: pd.DataFrame,
    accepted: np.ndarray,
    word_counts: np.ndarray,
    confidence_threshold: float,
    margin_threshold: float,
    min_words: int,
) -> dict[str, Any]:
    available = int(len(unlabeled_df))
    accepted = np.asarray(accepted, dtype=bool)
    return {
        "available_unlabeled": available,
        "selected": int(len(selected)),
        "coverage": float(len(selected) / available) if available else 0.0,
        "confidence_threshold": float(confidence_threshold),
        "margin_threshold": float(margin_threshold),
        "min_words": int(min_words),
        "rejected": int(available - len(selected)),
        "rejected_by_length": int((np.asarray(word_counts) < int(min_words)).sum()),
    }


def _json_safe_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_number(value) for key, value in row.items()}


def _json_safe_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _default_batch_size(device: str) -> int:
    return 8 if device == "cuda" else 1


def _default_gradient_accumulation(device: str) -> int:
    return 2 if device == "cuda" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = resolve_training_config(args)
    run_final_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
