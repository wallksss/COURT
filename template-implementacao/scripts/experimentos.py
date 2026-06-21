# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
# Nome:
# RA:
# ################################################################

"""Experiment orchestration for legal document classification."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import importlib.util
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared experiment settings."""

    text_col: str = "Body_clean"
    target_col: str = "Category"
    id_col: str = "Id"
    random_state: int = 42
    validation_size: float = 0.2
    cv_folds: int = 5
    scoring: str = "f1_macro"
    output_dir: str = "outputs"


def build_sparse_model_zoo(random_state: int = 42) -> dict[str, object]:
    """Create sparse text-classification pipelines.

    Models are deliberately heterogeneous:
    - word TF-IDF captures domain vocabulary and legal expressions;
    - char n-grams are robust to OCR noise and spelling variation;
    - word+char combines both views;
    - ComplementNB and LinearSVC are strong classical baselines for text.
    """

    _require_sklearn()
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    def word_tfidf():
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=120000,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def char_tfidf():
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.98,
            max_features=150000,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    return {
        "dummy_most_frequent": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=1000)),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "complement_nb_word_tfidf": Pipeline(
            [
                ("tfidf", word_tfidf()),
                ("clf", ComplementNB(alpha=0.5)),
            ]
        ),
        "logreg_word_tfidf": Pipeline(
            [
                ("tfidf", word_tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        C=3.0,
                        class_weight="balanced",
                        max_iter=3000,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svc_char_tfidf": Pipeline(
            [
                ("tfidf", char_tfidf()),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "sgd_logloss_word_tfidf": Pipeline(
            [
                ("tfidf", word_tfidf()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="elasticnet",
                        alpha=1e-5,
                        l1_ratio=0.15,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logreg_word_char_union": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", word_tfidf()),
                            ("char", char_tfidf()),
                        ]
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=3000,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_fast_sparse_model_zoo(random_state: int = 42) -> dict[str, object]:
    """Create a lightweight text model zoo suitable for CPU smoke/full runs.

    This is the default notebook path on machines without a GPU or without the
    advanced Transformer dependencies. It avoids the heavier char-ngram union so
    a "Run all" execution remains practical on ordinary laptops.
    """

    _require_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    def word_tfidf():
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    return {
        "complement_nb_fast": Pipeline(
            [
                ("tfidf", word_tfidf()),
                ("clf", ComplementNB(alpha=0.5)),
            ]
        ),
        "logreg_word_fast": Pipeline(
            [
                ("tfidf", word_tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=1200,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svc_word_fast": Pipeline(
            [
                ("tfidf", word_tfidf()),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "sgd_logloss_word_fast": Pipeline(
            [
                ("tfidf", word_tfidf()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="elasticnet",
                        alpha=1e-5,
                        l1_ratio=0.15,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_model_zoo(
    train_df: pd.DataFrame,
    models: dict[str, object] | None = None,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Train/evaluate all models with a stratified holdout split."""

    _require_sklearn()
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    config = config or ExperimentConfig()
    models = models or build_sparse_model_zoo(random_state=config.random_state)
    _validate_modeling_frame(train_df, config)

    X = train_df[config.text_col].fillna("").astype(str)
    y = train_df[config.target_col].astype(int)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    results = []
    fitted = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
        except Exception as exc:
            results.append(
                {
                    "model": name,
                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "f1_weighted": np.nan,
                    "precision_macro": np.nan,
                    "recall_macro": np.nan,
                    "n_train": len(X_train),
                    "n_valid": len(X_valid),
                    "error": repr(exc),
                }
            )
            continue
        row = {
            "model": name,
            "accuracy": accuracy_score(y_valid, y_pred),
            "f1_macro": f1_score(y_valid, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_valid, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_valid, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_valid, y_pred, average="macro", zero_division=0),
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "error": "",
        }
        results.append(row)
        fitted[name] = {
            "estimator": model,
            "y_valid": y_valid.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "classification_report": classification_report(y_valid, y_pred, output_dict=True, zero_division=0),
        }

    results_df = pd.DataFrame(results).sort_values(config.scoring, ascending=False, na_position="last").reset_index(drop=True)
    return results_df, fitted


def cross_validate_best_models(
    train_df: pd.DataFrame,
    model_names: Iterable[str] | None = None,
    models: dict[str, object] | None = None,
    config: ExperimentConfig | None = None,
) -> pd.DataFrame:
    """Run stratified cross-validation for selected models."""

    _require_sklearn()
    from sklearn.model_selection import StratifiedKFold, cross_validate

    config = config or ExperimentConfig()
    models = models or build_sparse_model_zoo(random_state=config.random_state)
    selected_names = list(model_names or models.keys())
    _validate_modeling_frame(train_df, config)

    X = train_df[config.text_col].fillna("").astype(str)
    y = train_df[config.target_col].astype(int)
    min_class = y.value_counts().min()
    n_splits = min(config.cv_folds, int(min_class))
    if n_splits < 2:
        raise ValueError("At least two samples per class are required for stratified CV.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    scoring = ["accuracy", "f1_macro", "f1_weighted"]
    rows = []
    for name in selected_names:
        scores = cross_validate(models[name], X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def fit_model_on_full_train(
    train_df: pd.DataFrame,
    model,
    config: ExperimentConfig | None = None,
):
    """Fit a selected estimator on all training data."""

    config = config or ExperimentConfig()
    _validate_modeling_frame(train_df, config)
    X = train_df[config.text_col].fillna("").astype(str)
    y = train_df[config.target_col].astype(int)
    model.fit(X, y)
    return model


def select_best_model(results_df: pd.DataFrame, fitted_models: dict[str, dict[str, object]], metric: str = "f1_macro"):
    """Return the fitted estimator with the best validation score."""

    if results_df.empty:
        raise ValueError("results_df is empty.")
    best_name = results_df.sort_values(metric, ascending=False).iloc[0]["model"]
    return best_name, fitted_models[best_name]["estimator"]


def generate_submission(
    model,
    test_df: pd.DataFrame,
    output_path: str | Path = "outputs/submission.csv",
    text_col: str = "Body_clean",
    id_col: str = "Id",
) -> pd.DataFrame:
    """Predict Kaggle categories and save a submission file."""

    if text_col not in test_df.columns:
        raise KeyError(f"Column '{text_col}' not found in test_df.")
    if id_col not in test_df.columns:
        raise KeyError(f"Column '{id_col}' not found in test_df.")

    predictions = model.predict(test_df[text_col].fillna("").astype(str))
    submission = pd.DataFrame({"Id": test_df[id_col].values, "Category": predictions.astype(int)})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def save_model(model, path: str | Path) -> Path:
    """Persist a fitted model with joblib."""

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install joblib to save models.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    """Load a persisted model with joblib."""

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install joblib to load models.") from exc

    return joblib.load(path)


def predict_with_confidence(model, texts: Iterable[str]) -> pd.DataFrame:
    """Predict labels and attach a confidence score when the estimator allows it.

    The function accepts scikit-learn pipelines/classifiers with either
    ``predict_proba`` or ``decision_function``. Probability-producing models are
    preferred because their confidence is easier to interpret. Margin-based
    estimators, such as LinearSVC, are converted with a softmax approximation.
    """

    X = pd.Series(list(texts)).fillna("").astype(str)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X))
        classes = np.asarray(getattr(model, "classes_", np.arange(scores.shape[1])))
        probabilities = scores
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        classes = np.asarray(getattr(model, "classes_", np.arange(scores.shape[1])))
        probabilities = _softmax(scores)
    else:
        predictions = np.asarray(model.predict(X)).astype(int)
        return pd.DataFrame(
            {
                "PredictedCategory": predictions,
                "Confidence": np.nan,
                "ConfidenceMargin": np.nan,
            }
        )

    best_indices = np.argmax(probabilities, axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        if probabilities.shape[1] > 1
        else sorted_probabilities[:, -1]
    )
    return pd.DataFrame(
        {
            "PredictedCategory": classes[best_indices].astype(int),
            "Confidence": probabilities[np.arange(len(best_indices)), best_indices],
            "ConfidenceMargin": margins,
        }
    )


def pseudo_label_unlabeled_samples(
    unlabeled_df: pd.DataFrame,
    model,
    config: ExperimentConfig | None = None,
    min_confidence: float = 0.90,
    text_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign pseudo-labels to unlabeled rows with conservative confidence gating."""

    config = config or ExperimentConfig()
    text_col = text_col or config.text_col
    if text_col not in unlabeled_df.columns:
        raise KeyError(f"Column '{text_col}' not found in unlabeled_df.")

    if unlabeled_df.empty:
        report = {
            "available_unlabeled": 0,
            "selected": 0,
            "coverage": 0.0,
            "min_confidence": min_confidence,
            "label_distribution": {},
        }
        return unlabeled_df.copy(), report

    confidence = predict_with_confidence(model, unlabeled_df[text_col])
    scored = unlabeled_df.reset_index(drop=True).copy()
    scored["PseudoCategory"] = confidence["PredictedCategory"].to_numpy()
    scored["PseudoConfidence"] = confidence["Confidence"].to_numpy()
    scored["PseudoConfidenceMargin"] = confidence["ConfidenceMargin"].to_numpy()
    scored["PseudoSource"] = f"model_confidence>={min_confidence:.2f}"

    selected = scored.loc[scored["PseudoConfidence"] >= min_confidence].copy()
    selected[config.target_col] = selected["PseudoCategory"].astype(int)
    report = {
        "available_unlabeled": int(len(unlabeled_df)),
        "selected": int(len(selected)),
        "coverage": float(len(selected) / len(unlabeled_df)) if len(unlabeled_df) else 0.0,
        "min_confidence": float(min_confidence),
        "label_distribution": selected[config.target_col].value_counts().sort_index().to_dict(),
    }
    return selected.reset_index(drop=True), report


def detect_label_issues(
    train_df: pd.DataFrame,
    model,
    config: ExperimentConfig | None = None,
    min_confidence: float = 0.90,
    text_col: str | None = None,
) -> pd.DataFrame:
    """Flag labeled rows whose current label disagrees with a confident model."""

    config = config or ExperimentConfig()
    text_col = text_col or config.text_col
    _validate_modeling_frame(train_df, ExperimentConfig(**{**config.__dict__, "text_col": text_col}))

    confidence = predict_with_confidence(model, train_df[text_col])
    audit = train_df.reset_index(drop=True).copy()
    audit["SuggestedCategory"] = confidence["PredictedCategory"].to_numpy()
    audit["SuggestedConfidence"] = confidence["Confidence"].to_numpy()
    audit["SuggestedConfidenceMargin"] = confidence["ConfidenceMargin"].to_numpy()

    issues = audit.loc[
        (audit[config.target_col].astype(int) != audit["SuggestedCategory"].astype(int))
        & (audit["SuggestedConfidence"] >= min_confidence)
    ].copy()

    ordered_columns = [
        col
        for col in [
            config.id_col,
            config.target_col,
            "SuggestedCategory",
            "SuggestedConfidence",
            "SuggestedConfidenceMargin",
            text_col,
        ]
        if col in issues.columns
    ]
    return issues[ordered_columns].sort_values("SuggestedConfidence", ascending=False).reset_index(drop=True)


def detect_label_issues_with_cv(
    train_df: pd.DataFrame,
    model,
    config: ExperimentConfig | None = None,
    min_confidence: float = 0.85,
    cv_folds: int | None = None,
) -> pd.DataFrame:
    """Audit noisy labels with out-of-fold predictions whenever possible."""

    _require_sklearn()
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    config = config or ExperimentConfig()
    _validate_modeling_frame(train_df, config)

    X = train_df[config.text_col].fillna("").astype(str)
    y = train_df[config.target_col].astype(int)
    min_class = y.value_counts().min()
    n_splits = min(int(cv_folds or config.cv_folds), int(min_class))
    if n_splits < 2:
        raise ValueError("At least two samples per class are required for label audit CV.")

    estimator = clone(model)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    scores = np.asarray(cross_val_predict(estimator, X, y, cv=cv, method=method, n_jobs=1))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    probabilities = scores if method == "predict_proba" else _softmax(scores)
    classes = np.sort(y.unique())
    best_indices = np.argmax(probabilities, axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    audit = train_df.reset_index(drop=True).copy()
    audit["SuggestedCategory"] = classes[best_indices].astype(int)
    audit["SuggestedConfidence"] = probabilities[np.arange(len(best_indices)), best_indices]
    audit["SuggestedConfidenceMargin"] = margins
    issues = audit.loc[
        (audit[config.target_col].astype(int) != audit["SuggestedCategory"].astype(int))
        & (audit["SuggestedConfidence"] >= min_confidence)
    ].copy()
    ordered_columns = [
        col
        for col in [
            config.id_col,
            config.target_col,
            "SuggestedCategory",
            "SuggestedConfidence",
            "SuggestedConfidenceMargin",
            config.text_col,
        ]
        if col in issues.columns
    ]
    return issues[ordered_columns].sort_values("SuggestedConfidence", ascending=False).reset_index(drop=True)


def build_training_set_with_pseudo_labels(
    labeled_df: pd.DataFrame,
    pseudo_labeled_df: pd.DataFrame | None = None,
    label_issues: pd.DataFrame | None = None,
    config: ExperimentConfig | None = None,
    correction_min_confidence: float | None = None,
) -> pd.DataFrame:
    """Build the final training frame from labels, pseudo-labels and safe corrections.

    Corrections are intentionally opt-in and confidence-gated. All corrected rows
    keep their original category in ``OriginalCategory`` for traceability.
    """

    config = config or ExperimentConfig()
    _validate_modeling_frame(labeled_df, config)
    training = labeled_df.copy()
    training["IsPseudoLabel"] = False
    training["OriginalCategory"] = np.nan
    training["LabelAuditAction"] = "kept"

    if (
        correction_min_confidence is not None
        and label_issues is not None
        and not label_issues.empty
        and config.id_col in training.columns
    ):
        confident_issues = label_issues.loc[
            label_issues["SuggestedConfidence"] >= correction_min_confidence
        ].copy()
        corrections = confident_issues.set_index(config.id_col)["SuggestedCategory"].to_dict()
        correction_mask = training[config.id_col].isin(corrections)
        training.loc[correction_mask, "OriginalCategory"] = training.loc[correction_mask, config.target_col]
        training.loc[correction_mask, config.target_col] = (
            training.loc[correction_mask, config.id_col].map(corrections).astype(int)
        )
        training.loc[correction_mask, "LabelAuditAction"] = "auto_corrected"

    if pseudo_labeled_df is not None and not pseudo_labeled_df.empty:
        pseudo = pseudo_labeled_df.copy()
        if config.target_col not in pseudo.columns:
            raise KeyError(f"Column '{config.target_col}' not found in pseudo_labeled_df.")
        pseudo["IsPseudoLabel"] = True
        pseudo["OriginalCategory"] = np.nan
        pseudo["LabelAuditAction"] = "pseudo_labeled"
        training = pd.concat([training, pseudo], ignore_index=True, sort=False)

    training[config.target_col] = training[config.target_col].astype(int)
    return training.reset_index(drop=True)


def transformer_candidate_models() -> pd.DataFrame:
    """Catalog of transformer/backbone candidates for this problem."""

    rows = [
        {
            "model_id": "neuralmind/bert-base-portuguese-cased",
            "family": "BERTimbau",
            "why": "Strong Brazilian Portuguese encoder baseline.",
            "max_length": 512,
        },
        {
            "model_id": "neuralmind/bert-large-portuguese-cased",
            "family": "BERTimbau Large",
            "why": "Larger Portuguese encoder when GPU memory allows.",
            "max_length": 512,
        },
        {
            "model_id": "PORTULAN/albertina-ptbr-base",
            "family": "Albertina PT-BR",
            "why": "Open Portuguese DeBERTa-style encoder trained for PT-BR.",
            "max_length": 512,
        },
        {
            "model_id": "answerdotai/ModernBERT-base",
            "family": "ModernBERT",
            "why": "Efficient long-context encoder; useful for long legal pages.",
            "max_length": 8192,
        },
        {
            "model_id": "allenai/longformer-base-4096",
            "family": "Longformer",
            "why": "Long-document attention baseline for documents above 512 tokens.",
            "max_length": 4096,
        },
    ]
    return pd.DataFrame(rows)


def gpu_environment_report() -> dict[str, object]:
    """Return a compact report about accelerator availability.

    This function is safe on CPU-only machines and useful before starting the
    advanced notebook cells on a dedicated NVIDIA server.
    """

    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "preferred_device": "cpu",
            "cuda_available": False,
            "mps_available": False,
            "message": "Install torch first to run Transformer experiments.",
        }

    report = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "preferred_device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "devices": [],
        "fp16_recommended": bool(torch.cuda.is_available()),
        "bf16_supported": False,
    }
    if torch.cuda.is_available():
        report["preferred_device"] = "cuda"
        report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            report["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    elif report["mps_available"]:
        report["preferred_device"] = "mps"
        report["devices"].append(
            {
                "index": 0,
                "name": "Apple Metal Performance Shaders",
                "total_memory_gb": None,
                "capability": "mps",
            }
        )
    return report


def auto_runtime_profile(prefer_transformer: bool = True) -> dict[str, object]:
    """Choose a runnable training profile for the current machine.

    The goal is to let the notebook run without manual flags:
    - every machine runs the classical baselines;
    - embeddings and Transformer fine-tuning are enabled regardless of device;
    - batch size and precision are adjusted for CUDA, MPS or CPU.
    """

    env = gpu_environment_report()
    device = env.get("preferred_device", "cpu")
    advanced_modules = ["torch", "transformers", "datasets"]
    advanced_available = all(importlib.util.find_spec(module) is not None for module in advanced_modules)
    run_transformer = bool(prefer_transformer)

    cuda_memory = 0.0
    if device == "cuda" and env.get("devices"):
        cuda_memory = float(env["devices"][0].get("total_memory_gb") or 0.0)

    if device == "cuda":
        batch_size = 8 if cuda_memory >= 12 else 4
        max_length = 512
        use_lora = False
        quantization_4bit = False
    elif device == "mps":
        batch_size = 2
        max_length = 512
        use_lora = False
        quantization_4bit = False
    else:
        batch_size = 1
        max_length = 512
        use_lora = False
        quantization_4bit = False

    return {
        "device": device,
        "environment": env,
        "advanced_available": advanced_available,
        "run_classical_baselines": True,
        "classical_model_zoo": "full",
        "run_cross_validation": True,
        "run_transformer": run_transformer,
        "run_embeddings": True,
        "recommended_transformer": "neuralmind/bert-base-portuguese-cased",
        "transformer_batch_size": batch_size,
        "transformer_max_length": max_length,
        "transformer_epochs": 3,
        "transformer_learning_rate": 2e-5,
        "use_lora": use_lora,
        "quantization_4bit": quantization_4bit,
        "debug_sample_size": None,
        "notes": _runtime_notes(device, advanced_available, run_transformer),
    }


def _runtime_notes(device: str, advanced_available: bool, run_transformer: bool) -> str:
    if run_transformer and advanced_available:
        return f"Complete pipeline enabled automatically on {device}."
    if run_transformer and not advanced_available:
        return "Complete pipeline requested; install advanced dependencies before Transformer/embedding stages."
    return f"Classical pipeline enabled on {device}."


def _split_text_into_word_chunks(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Split long documents into overlapping word chunks."""

    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words.")

    words = str(text or "").split()
    if not words:
        return [""]
    if len(words) <= max_words:
        return [" ".join(words)]

    step = max_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + max_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + max_words >= len(words):
            break
    return chunks


def build_sentence_embedding_matrix(
    texts: Iterable[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 16,
    normalize_embeddings: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Encode texts with SentenceTransformers when the dependency is available."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers to run embedding experiments.") from exc

    device = device or gpu_environment_report().get("preferred_device", "cpu")
    model = SentenceTransformer(model_name, device=device)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )


def build_chunked_sentence_embedding_matrix(
    texts: Iterable[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 16,
    normalize_embeddings: bool = True,
    device: str | None = None,
    max_words: int = 220,
    overlap_words: int = 40,
    pooling: str = "mean",
) -> np.ndarray:
    """Encode long documents by chunking text and pooling chunk embeddings.

    Sentence embedding models usually receive a limited token window. Legal
    documents frequently exceed that window, so full-text encoding may represent
    only the beginning of each document. Chunking keeps later legal signals in
    the representation while preserving a simple dense matrix interface.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers to run embedding experiments.") from exc

    if pooling not in {"mean", "max"}:
        raise ValueError("pooling must be either 'mean' or 'max'.")

    documents = list(texts)
    chunks: list[str] = []
    doc_indices: list[int] = []
    for doc_idx, text in enumerate(documents):
        doc_chunks = _split_text_into_word_chunks(text, max_words=max_words, overlap_words=overlap_words)
        chunks.extend(doc_chunks)
        doc_indices.extend([doc_idx] * len(doc_chunks))

    device = device or gpu_environment_report().get("preferred_device", "cpu")
    model = SentenceTransformer(model_name, device=device)
    chunk_embeddings = np.asarray(
        model.encode(
            chunks,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=True,
        )
    )
    doc_indices_array = np.asarray(doc_indices)
    doc_embeddings = np.zeros((len(documents), chunk_embeddings.shape[1]), dtype=chunk_embeddings.dtype)

    if pooling == "mean":
        counts = np.zeros(len(documents), dtype=np.float32)
        for doc_idx, embedding in zip(doc_indices_array, chunk_embeddings):
            doc_embeddings[doc_idx] += embedding
            counts[doc_idx] += 1
        doc_embeddings = doc_embeddings / np.maximum(counts[:, None], 1.0)
    else:
        doc_embeddings[:] = -np.inf
        for doc_idx, embedding in zip(doc_indices_array, chunk_embeddings):
            doc_embeddings[doc_idx] = np.maximum(doc_embeddings[doc_idx], embedding)
        doc_embeddings[~np.isfinite(doc_embeddings)] = 0.0

    if normalize_embeddings:
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_embeddings = doc_embeddings / np.maximum(norms, 1e-12)
    return doc_embeddings


def evaluate_embedding_classifier(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate linear classifiers over dense sentence/document embeddings."""

    _require_sklearn()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC

    config = config or ExperimentConfig()
    y = train_df[config.target_col].astype(int)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        embeddings,
        y,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    models = {
        "embedding_logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=config.random_state),
        "embedding_linear_svc": LinearSVC(class_weight="balanced", random_state=config.random_state),
    }
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_valid, y_pred),
                "f1_macro": f1_score(y_valid, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_valid, y_pred, average="weighted", zero_division=0),
            }
        )
        fitted[name] = model
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False), fitted


def build_embedding_classifier_zoo(random_state: int = 42) -> dict[str, object]:
    """Create classifiers for dense embedding matrices."""

    _require_sklearn()
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    return {
        "embedding_logreg": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "embedding_linear_svc": LinearSVC(
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def fit_embedding_classifier_on_full_train(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str = "embedding_logreg",
    config: ExperimentConfig | None = None,
):
    """Fit one dense-embedding classifier on all labeled data."""

    config = config or ExperimentConfig()
    models = build_embedding_classifier_zoo(random_state=config.random_state)
    if model_name not in models:
        raise KeyError(f"Unknown embedding classifier: {model_name}")
    model = models[model_name]
    y = train_df[config.target_col].astype(int)
    model.fit(embeddings, y)
    return model


def generate_embedding_submission(
    model,
    test_embeddings: np.ndarray,
    test_ids: Iterable,
    output_path: str | Path = "outputs/submission_embeddings.csv",
) -> pd.DataFrame:
    """Generate a Kaggle submission from a dense-embedding classifier."""

    predictions = model.predict(test_embeddings).astype(int)
    submission = pd.DataFrame({"Id": list(test_ids), "Category": predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def _instantiate_trainer_compat(trainer_cls, trainer_kwargs: dict[str, object], tokenizer, **extra_kwargs):
    """Instantiate HuggingFace Trainer across tokenizer/processing_class API changes."""

    kwargs = dict(trainer_kwargs)
    kwargs.pop("tokenizer", None)
    try:
        signature = inspect.signature(trainer_cls.__init__)
        supports_processing_class = "processing_class" in signature.parameters
    except (TypeError, ValueError):
        supports_processing_class = False

    if supports_processing_class:
        kwargs["processing_class"] = tokenizer

    trainer = trainer_cls(**extra_kwargs, **kwargs)
    if not hasattr(trainer, "processing_class") or getattr(trainer, "processing_class", None) is None:
        trainer.processing_class = tokenizer
    return trainer


def fine_tune_transformer_classifier(
    train_df: pd.DataFrame,
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    config: ExperimentConfig | None = None,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    max_length: int = 512,
    output_dir: str = "outputs/transformer",
    weight_decay: float = 0.01,
    use_class_weights: bool = True,
    use_lora: bool = False,
    quantization_4bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    device: str | None = None,
):
    """Fine-tune a HuggingFace sequence classifier.

    This optional routine is intentionally isolated from the sparse benchmark so
    the project remains runnable on CPU with scikit-learn. It requires
    transformers, datasets, evaluate and torch.
    """

    try:
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Install transformers, datasets, torch and scikit-learn to fine-tune transformers.") from exc

    import torch

    config = config or ExperimentConfig()
    _validate_modeling_frame(train_df, config)
    device = device or gpu_environment_report().get("preferred_device", "cpu")

    data = train_df[[config.text_col, config.target_col]].rename(columns={config.text_col: "text", config.target_col: "label"})
    train_split, valid_split = train_test_split(
        data,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=data["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_kwargs = {
        "num_labels": 5,
        "id2label": {idx: str(idx) for idx in range(5)},
        "label2id": {str(idx): idx for idx in range(5)},
    }
    if quantization_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA/NVIDIA environment.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("Install bitsandbytes and a recent transformers version for QLoRA.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError("Install peft to use LoRA/QLoRA.") from exc
        if quantization_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or ["query", "value"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    if not quantization_4bit:
        model.to(torch.device(device))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_dataset = Dataset.from_pandas(train_split.reset_index(drop=True)).map(tokenize, batched=True)
    valid_dataset = Dataset.from_pandas(valid_split.reset_index(drop=True)).map(tokenize, batched=True)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels", None)
            if labels is None:
                labels = inputs.pop("label", None)
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if labels is None:
                loss = outputs.get("loss")
                return (loss, outputs) if return_outputs else loss
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    class_weights = None
    if use_class_weights:
        counts = train_split["label"].value_counts().sort_index()
        weights = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor([weights.get(idx, 1.0) for idx in range(5)], dtype=torch.float)

    args_kwargs = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_train_batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "save_strategy": "epoch",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "report_to": "none",
        "fp16": bool(device == "cuda" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        "bf16": bool(device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    if device == "cpu":
        args_kwargs["no_cuda"] = True
    elif device == "mps":
        args_kwargs["use_mps_device"] = True
    try:
        args = TrainingArguments(evaluation_strategy="epoch", **args_kwargs)
    except TypeError:
        try:
            args = TrainingArguments(eval_strategy="epoch", **args_kwargs)
        except TypeError:
            args_kwargs.pop("use_mps_device", None)
            args = TrainingArguments(eval_strategy="epoch", **args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    if use_class_weights:
        trainer = _instantiate_trainer_compat(
            WeightedTrainer,
            trainer_kwargs=trainer_kwargs,
            tokenizer=tokenizer,
            class_weights=class_weights,
        )
    else:
        trainer = _instantiate_trainer_compat(
            Trainer,
            trainer_kwargs=trainer_kwargs,
            tokenizer=tokenizer,
        )
    trainer.train()
    return trainer


def generate_transformer_submission(
    trainer,
    test_df: pd.DataFrame,
    tokenizer=None,
    text_col: str = "Body_clean",
    id_col: str = "Id",
    output_path: str | Path = "outputs/submission_transformer.csv",
    max_length: int = 512,
) -> pd.DataFrame:
    """Generate a Kaggle submission with a fine-tuned HuggingFace Trainer."""

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install datasets to generate transformer submissions.") from exc

    if text_col not in test_df.columns:
        raise KeyError(f"Column '{text_col}' not found in test_df.")
    if id_col not in test_df.columns:
        raise KeyError(f"Column '{id_col}' not found in test_df.")

    tokenizer = tokenizer or getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("A tokenizer or trainer.processing_class is required.")

    dataset = Dataset.from_pandas(test_df[[id_col, text_col]].rename(columns={text_col: "text"}).reset_index(drop=True))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    predictions = trainer.predict(tokenized)
    labels = np.argmax(predictions.predictions, axis=-1).astype(int)
    submission = pd.DataFrame({"Id": test_df[id_col].values, "Category": labels})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def _validate_modeling_frame(df: pd.DataFrame, config: ExperimentConfig) -> None:
    missing = [col for col in (config.text_col, config.target_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for modeling: {missing}")
    if df[config.target_col].nunique() < 2:
        raise ValueError("At least two classes are required for modeling.")


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install scikit-learn to run experiments.") from exc


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
