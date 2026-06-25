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

"""Experiment orchestration for legal document classification."""

from __future__ import annotations

import inspect
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            max_features=200000,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def char_tfidf():
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.98,
            max_features=200000,
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
        "linear_svc_calibrated_word_char_union": Pipeline(
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
                    _make_calibrated_linear_svc(random_state=random_state),
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


def _make_calibrated_linear_svc(random_state: int = 42):
    """Create a probability-capable LinearSVC across scikit-learn versions."""

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    svc = LinearSVC(C=1.0, class_weight="balanced", random_state=random_state)
    try:
        return CalibratedClassifierCV(estimator=svc, cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=svc, cv=3)


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


def make_validation_splitters(
    train_df: pd.DataFrame,
    config: ExperimentConfig | None = None,
    text_col: str | None = None,
) -> dict[str, object]:
    """Create the three validation schemes requested by the CSV-only plan."""

    _require_sklearn()
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    config = config or ExperimentConfig()
    text_col = text_col or config.text_col
    _validate_modeling_frame(train_df, ExperimentConfig(**{**config.__dict__, "text_col": text_col}))

    y = train_df[config.target_col].astype(int)
    min_class = int(y.value_counts().min())
    stratified_splits = min(config.cv_folds, min_class)
    if stratified_splits < 2:
        raise ValueError("At least two samples per class are required for stratified CV.")

    n_groups = train_df[text_col].fillna("").astype(str).nunique()
    group_splits = min(config.cv_folds, int(n_groups))
    if group_splits < 2:
        raise ValueError("At least two unique texts are required for GroupKFold.")

    return {
        "stratified": StratifiedKFold(
            n_splits=stratified_splits,
            shuffle=True,
            random_state=config.random_state,
        ),
        "group_by_text": GroupKFold(n_splits=group_splits),
        "sequential": list(make_sequential_validation_folds(train_df, config=config)),
    }


def make_sequential_validation_folds(
    train_df: pd.DataFrame,
    config: ExperimentConfig | None = None,
    n_splits: int | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield contiguous validation folds after sorting by Id."""

    config = config or ExperimentConfig()
    if config.id_col not in train_df.columns:
        raise KeyError(f"Column '{config.id_col}' not found.")
    _validate_modeling_frame(train_df, config)

    ordered_positions = np.argsort(train_df[config.id_col].to_numpy())
    fold_count = int(n_splits or config.cv_folds)
    fold_count = max(2, min(fold_count, len(ordered_positions)))
    for valid_idx in np.array_split(ordered_positions, fold_count):
        train_idx = np.setdiff1d(np.arange(len(train_df)), valid_idx, assume_unique=False)
        yield train_idx, np.asarray(valid_idx)


def make_oof_probabilities(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    classes: Sequence[int] | None = None,
    config: ExperimentConfig | None = None,
    cv_folds: int | None = None,
    random_state: int | None = None,
    groups: Sequence[object] | None = None,
    refit_full_for_test: bool = True,
    output_oof_path: str | Path | None = None,
    output_test_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate leakage-safe OOF probabilities and final test probabilities.

    OOF rows always come from models that did not see those labels. Test
    probabilities are produced by a full-train refit by default, so final
    submissions use all available labeled information after validation choices
    are made. Set ``refit_full_for_test=False`` to average fold models instead.
    """

    _require_sklearn()
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    config = config or ExperimentConfig()
    if random_state is not None:
        config = ExperimentConfig(**{**config.__dict__, "random_state": int(random_state)})
    _validate_modeling_frame(train_df, config)
    if config.text_col not in test_df.columns:
        raise KeyError(f"Column '{config.text_col}' not found in test_df.")

    X = train_df[config.text_col].fillna("").astype(str).reset_index(drop=True)
    y = train_df[config.target_col].astype(int).reset_index(drop=True)
    X_test = test_df[config.text_col].fillna("").astype(str).reset_index(drop=True)
    class_values = np.asarray(sorted(classes if classes is not None else y.unique()))

    if groups is None:
        min_class = int(y.value_counts().min())
        n_splits = min(int(cv_folds or config.cv_folds), min_class)
        if n_splits < 2:
            raise ValueError("At least two samples per class are required for OOF CV.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
        split_iter = splitter.split(X, y)
    else:
        group_values = pd.Series(groups).reset_index(drop=True)
        n_groups = int(group_values.nunique())
        n_splits = min(int(cv_folds or config.cv_folds), n_groups)
        if n_splits < 2:
            raise ValueError("At least two groups are required for grouped OOF CV.")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, group_values)

    oof_probs = np.zeros((len(train_df), len(class_values)), dtype=float)
    test_probs = np.zeros((len(test_df), len(class_values)), dtype=float)
    fold_count = 0
    for train_idx, valid_idx in split_iter:
        estimator = clone(model)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_probs[valid_idx] = _predict_proba_aligned(estimator, X.iloc[valid_idx], class_values)
        test_probs += _predict_proba_aligned(estimator, X_test, class_values)
        fold_count += 1

    if fold_count == 0:
        raise ValueError("No CV folds were produced.")
    if refit_full_for_test:
        final_estimator = clone(model)
        final_estimator.fit(X, y)
        test_probs = _predict_proba_aligned(final_estimator, X_test, class_values)
    else:
        test_probs /= fold_count
    oof_probs = _normalize_probabilities(oof_probs)
    test_probs = _normalize_probabilities(test_probs)

    if output_oof_path is not None:
        output_oof_path = Path(output_oof_path)
        output_oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_oof_path, oof_probs)
    if output_test_path is not None:
        output_test_path = Path(output_test_path)
        output_test_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_test_path, test_probs)
    return oof_probs, test_probs


def average_probabilities(
    probability_sets: dict[str, np.ndarray] | Sequence[np.ndarray],
    weights: dict[str, float] | Sequence[float] | None = None,
) -> np.ndarray:
    """Average model probabilities with optional non-negative weights."""

    if isinstance(probability_sets, dict):
        names = list(probability_sets)
        arrays = [np.asarray(probability_sets[name], dtype=float) for name in names]
        if weights is None:
            weight_values = np.ones(len(names), dtype=float)
        elif isinstance(weights, dict):
            weight_values = np.asarray([weights[name] for name in names], dtype=float)
        else:
            weight_values = np.asarray(list(weights), dtype=float)
    else:
        arrays = [np.asarray(values, dtype=float) for values in probability_sets]
        weight_values = np.ones(len(arrays), dtype=float) if weights is None else np.asarray(list(weights), dtype=float)

    if not arrays:
        raise ValueError("At least one probability array is required.")
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        raise ValueError("All probability arrays must have the same shape.")
    if np.any(weight_values < 0):
        raise ValueError("Ensemble weights must be non-negative.")
    if not np.isfinite(weight_values).all() or weight_values.sum() <= 0:
        raise ValueError("Ensemble weights must sum to a positive finite value.")

    weight_values = weight_values / weight_values.sum()
    combined = np.zeros(first_shape, dtype=float)
    for weight, values in zip(weight_values, arrays):
        combined += weight * _normalize_probabilities(values)
    return _normalize_probabilities(combined)


def optimize_ensemble_weights(
    probability_sets: dict[str, np.ndarray],
    y_true: Sequence[int],
    classes: Sequence[int],
    step: float = 0.05,
) -> dict[str, object]:
    """Grid-search convex ensemble weights for macro F1."""

    _require_sklearn()
    from sklearn.metrics import f1_score

    if step <= 0 or step > 1:
        raise ValueError("step must be in the interval (0, 1].")
    names = list(probability_sets)
    if not names:
        raise ValueError("probability_sets cannot be empty.")

    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must evenly divide 1.0, for example 0.05 or 0.10.")

    y_true = np.asarray(y_true).astype(int)
    class_values = np.asarray(classes).astype(int)
    best: dict[str, object] | None = None
    for allocation in product(range(units + 1), repeat=len(names)):
        if sum(allocation) != units:
            continue
        weights = {name: allocation[idx] / units for idx, name in enumerate(names)}
        combined = average_probabilities(probability_sets, weights=weights)
        predictions = class_values[np.argmax(combined, axis=1)]
        score = f1_score(y_true, predictions, average="macro", zero_division=0)
        if best is None or score > best["f1_macro"]:
            best = {
                "weights": weights,
                "f1_macro": float(score),
                "probabilities": combined,
                "predictions": predictions,
            }
    if best is None:
        raise RuntimeError("No ensemble weight combination was evaluated.")
    return best


def duplicate_prior_probabilities(
    reference_df: pd.DataFrame,
    target_texts: Iterable[object],
    classes: Sequence[int],
    text_col: str = "Body_clean",
    target_col: str = "Category",
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute smoothed p(label | exact cleaned text) from reference rows only."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if text_col not in reference_df.columns:
        raise KeyError(f"Column '{text_col}' not found.")
    if target_col not in reference_df.columns:
        raise KeyError(f"Column '{target_col}' not found.")

    class_values = np.asarray(classes).astype(int)
    class_to_col = {label: idx for idx, label in enumerate(class_values)}
    uniform = np.ones(len(class_values), dtype=float) / len(class_values)
    counts_by_text: dict[str, np.ndarray] = {}
    for text, label in zip(reference_df[text_col].fillna("").astype(str), reference_df[target_col].astype(int)):
        if label not in class_to_col:
            continue
        counts_by_text.setdefault(text, np.zeros(len(class_values), dtype=float))[class_to_col[label]] += 1.0

    rows = []
    for text in pd.Series(list(target_texts)).fillna("").astype(str):
        counts = counts_by_text.get(text)
        if counts is None:
            rows.append(uniform.copy())
            continue
        probs = (counts + alpha) / (counts.sum() + alpha * len(class_values))
        rows.append(probs)
    return np.vstack(rows) if rows else np.zeros((0, len(class_values)), dtype=float)


def make_oof_duplicate_prior_probabilities(
    train_df: pd.DataFrame,
    classes: Sequence[int],
    config: ExperimentConfig | None = None,
    cv_folds: int | None = None,
    groups: Sequence[object] | None = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """Generate duplicate priors for train rows without using their own labels."""

    _require_sklearn()
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    config = config or ExperimentConfig()
    _validate_modeling_frame(train_df, config)
    X = train_df[config.text_col].fillna("").astype(str).reset_index(drop=True)
    y = train_df[config.target_col].astype(int).reset_index(drop=True)
    class_values = np.asarray(classes).astype(int)
    priors = np.zeros((len(train_df), len(class_values)), dtype=float)

    if groups is None:
        min_class = int(y.value_counts().min())
        n_splits = min(int(cv_folds or config.cv_folds), min_class)
        if n_splits < 2:
            raise ValueError("At least two samples per class are required for OOF duplicate priors.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
        split_iter = splitter.split(X, y)
    else:
        group_values = pd.Series(groups).reset_index(drop=True)
        n_splits = min(int(cv_folds or config.cv_folds), int(group_values.nunique()))
        if n_splits < 2:
            raise ValueError("At least two groups are required for OOF duplicate priors.")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, group_values)

    for train_idx, valid_idx in split_iter:
        reference = train_df.iloc[train_idx]
        priors[valid_idx] = duplicate_prior_probabilities(
            reference,
            target_texts=X.iloc[valid_idx],
            classes=class_values,
            text_col=config.text_col,
            target_col=config.target_col,
            alpha=alpha,
        )
    return _normalize_probabilities(priors)


def combine_with_duplicate_prior(
    text_probs: np.ndarray,
    duplicate_probs: np.ndarray,
    lambda_dup: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Combine textual probabilities with duplicate priors in log space."""

    text_probs = _normalize_probabilities(np.asarray(text_probs, dtype=float))
    duplicate_probs = _normalize_probabilities(np.asarray(duplicate_probs, dtype=float))
    if text_probs.shape != duplicate_probs.shape:
        raise ValueError("text_probs and duplicate_probs must have the same shape.")
    log_scores = np.log(text_probs + eps) + float(lambda_dup) * np.log(duplicate_probs + eps)
    return _softmax(log_scores)


def optimize_duplicate_lambda(
    text_probs: np.ndarray,
    duplicate_probs: np.ndarray,
    y_true: Sequence[int],
    classes: Sequence[int],
    lambda_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
) -> dict[str, object]:
    """Select the duplicate-prior strength that maximizes macro F1."""

    _require_sklearn()
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(int)
    class_values = np.asarray(classes).astype(int)
    best = None
    for lambda_dup in lambda_grid:
        combined = combine_with_duplicate_prior(text_probs, duplicate_probs, lambda_dup=lambda_dup)
        predictions = class_values[np.argmax(combined, axis=1)]
        score = f1_score(y_true, predictions, average="macro", zero_division=0)
        candidate = {
            "lambda_dup": float(lambda_dup),
            "f1_macro": float(score),
            "probabilities": combined,
            "predictions": predictions,
        }
        if best is None or candidate["f1_macro"] > best["f1_macro"]:
            best = candidate
    return best


def estimate_transition_matrix(
    labels: Sequence[int],
    classes: Sequence[int] | None = None,
    ids: Sequence[object] | None = None,
    alpha: float = 1.0,
    max_id_gap: float | None = None,
) -> np.ndarray:
    """Estimate a smoothed transition matrix P(y_i | y_{i-1})."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    labels_array = np.asarray(labels).astype(int)
    if classes is None:
        class_values = np.asarray(sorted(np.unique(labels_array))).astype(int)
    else:
        class_values = np.asarray(classes).astype(int)
    class_to_idx = {label: idx for idx, label in enumerate(class_values)}

    if ids is not None:
        order = np.argsort(np.asarray(ids))
        labels_array = labels_array[order]
        ids_array = np.asarray(ids)[order]
    else:
        ids_array = None

    counts = np.full((len(class_values), len(class_values)), float(alpha), dtype=float)
    for pos in range(1, len(labels_array)):
        if ids_array is not None and max_id_gap is not None:
            try:
                if float(ids_array[pos]) - float(ids_array[pos - 1]) > max_id_gap:
                    continue
            except (TypeError, ValueError):
                pass
        prev_label = labels_array[pos - 1]
        next_label = labels_array[pos]
        if prev_label in class_to_idx and next_label in class_to_idx:
            counts[class_to_idx[prev_label], class_to_idx[next_label]] += 1.0
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)


def sequential_signal_report(
    train_df: pd.DataFrame,
    classes: Sequence[int] | None = None,
    config: ExperimentConfig | None = None,
) -> dict[str, object]:
    """Measure how much adjacent Id order agrees with labels."""

    config = config or ExperimentConfig()
    if config.id_col not in train_df.columns:
        raise KeyError(f"Column '{config.id_col}' not found.")
    if config.target_col not in train_df.columns:
        raise KeyError(f"Column '{config.target_col}' not found.")
    ordered = train_df.sort_values(config.id_col).reset_index(drop=True)
    labels = ordered[config.target_col].astype(int).to_numpy()
    id_diff = ordered[config.id_col].diff().dropna()
    transition = estimate_transition_matrix(
        labels,
        classes=classes,
        ids=ordered[config.id_col].to_numpy(),
    )
    same_next = float((labels[1:] == labels[:-1]).mean()) if len(labels) > 1 else np.nan
    return {
        "same_next": same_next,
        "id_diff_describe": id_diff.describe().to_dict(),
        "transition_matrix": transition,
    }


def viterbi_decode(
    emission_probs: np.ndarray,
    transition_probs: np.ndarray,
    allowed_label_indices: Sequence[set[int] | None] | None = None,
    lambda_transition: float = 1.0,
    start_probs: np.ndarray | None = None,
    eps: float = 1e-12,
) -> list[int]:
    """Decode the best label-index sequence with optional fixed positions."""

    emissions = _normalize_probabilities(np.asarray(emission_probs, dtype=float))
    transitions = _normalize_probabilities(np.asarray(transition_probs, dtype=float))
    if emissions.ndim != 2:
        raise ValueError("emission_probs must be a 2D array.")
    n_positions, n_classes = emissions.shape
    if transitions.shape != (n_classes, n_classes):
        raise ValueError("transition_probs must have shape (n_classes, n_classes).")
    if allowed_label_indices is not None and len(allowed_label_indices) != n_positions:
        raise ValueError("allowed_label_indices length must match emission rows.")

    log_emissions = np.log(emissions + eps)
    log_transitions = np.log(transitions + eps) * float(lambda_transition)
    if start_probs is None:
        log_start = np.full(n_classes, -np.log(n_classes), dtype=float)
    else:
        log_start = np.log(_normalize_probabilities(np.asarray(start_probs, dtype=float).reshape(1, -1))[0] + eps)

    dp = np.full((n_positions, n_classes), -np.inf, dtype=float)
    back = np.zeros((n_positions, n_classes), dtype=int)
    dp[0] = log_start + log_emissions[0] + _allowed_mask(n_classes, None if allowed_label_indices is None else allowed_label_indices[0])
    for pos in range(1, n_positions):
        mask = _allowed_mask(n_classes, None if allowed_label_indices is None else allowed_label_indices[pos])
        scores = dp[pos - 1][:, None] + log_transitions
        back[pos] = np.argmax(scores, axis=0)
        dp[pos] = scores[back[pos], np.arange(n_classes)] + log_emissions[pos] + mask

    decoded = [int(np.argmax(dp[-1]))]
    for pos in range(n_positions - 1, 0, -1):
        decoded.append(int(back[pos, decoded[-1]]))
    decoded.reverse()
    return decoded


def transductive_viterbi_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_probs: np.ndarray,
    classes: Sequence[int],
    config: ExperimentConfig | None = None,
    transition_probs: np.ndarray | None = None,
    lambda_transition: float = 1.0,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Infer test labels in Id order while keeping training labels fixed."""

    config = config or ExperimentConfig()
    for frame_name, frame in [("train_df", train_df), ("test_df", test_df)]:
        if config.id_col not in frame.columns:
            raise KeyError(f"Column '{config.id_col}' not found in {frame_name}.")
    if config.target_col not in train_df.columns:
        raise KeyError(f"Column '{config.target_col}' not found in train_df.")

    class_values = np.asarray(classes).astype(int)
    class_to_idx = {label: idx for idx, label in enumerate(class_values)}
    test_probs = _normalize_probabilities(np.asarray(test_probs, dtype=float))
    if test_probs.shape != (len(test_df), len(class_values)):
        raise ValueError("test_probs shape must be (len(test_df), n_classes).")

    if transition_probs is None:
        transition_probs = estimate_transition_matrix(
            train_df[config.target_col].astype(int),
            classes=class_values,
            ids=train_df[config.id_col],
        )

    train_part = pd.DataFrame(
        {
            config.id_col: train_df[config.id_col].to_numpy(),
            "_source": "train",
            "_row": np.arange(len(train_df)),
        }
    )
    test_part = pd.DataFrame(
        {
            config.id_col: test_df[config.id_col].to_numpy(),
            "_source": "test",
            "_row": np.arange(len(test_df)),
            "_test_order": np.arange(len(test_df)),
        }
    )
    all_data = pd.concat([train_part, test_part], ignore_index=True).sort_values(config.id_col).reset_index(drop=True)

    emissions = np.full((len(all_data), len(class_values)), 1.0 / len(class_values), dtype=float)
    allowed: list[set[int] | None] = []
    train_labels = train_df[config.target_col].astype(int).reset_index(drop=True)
    for pos, row in all_data.iterrows():
        if row["_source"] == "train":
            label = int(train_labels.iloc[int(row["_row"])])
            if label not in class_to_idx:
                raise ValueError(f"Training label {label} is not present in classes.")
            allowed.append({class_to_idx[label]})
        else:
            emissions[pos] = test_probs[int(row["_row"])]
            allowed.append(None)

    decoded_indices = viterbi_decode(
        emissions,
        transition_probs,
        allowed_label_indices=allowed,
        lambda_transition=lambda_transition,
    )
    all_data["Category"] = class_values[np.asarray(decoded_indices)].astype(int)
    submission = (
        all_data.loc[all_data["_source"] == "test", [config.id_col, "Category", "_test_order"]]
        .sort_values("_test_order")
        .drop(columns="_test_order")
        .reset_index(drop=True)
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
    return submission


def optimize_transition_lambda_oof(
    train_df: pd.DataFrame,
    emission_probs: np.ndarray,
    classes: Sequence[int],
    config: ExperimentConfig | None = None,
    lambda_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    alpha_transition: float = 1.0,
    folds: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, object]:
    """Validate transductive Viterbi by hiding labels inside train folds."""

    _require_sklearn()
    from sklearn.metrics import f1_score

    config = config or ExperimentConfig()
    _validate_modeling_frame(train_df, config)
    if config.id_col not in train_df.columns:
        raise KeyError(f"Column '{config.id_col}' not found.")

    class_values = np.asarray(classes).astype(int)
    class_to_idx = {label: idx for idx, label in enumerate(class_values)}
    emissions = _normalize_probabilities(np.asarray(emission_probs, dtype=float))
    if emissions.shape != (len(train_df), len(class_values)):
        raise ValueError("emission_probs shape must be (len(train_df), n_classes).")

    fold_list = list(folds) if folds is not None else list(make_sequential_validation_folds(train_df, config=config))
    y_true = train_df[config.target_col].astype(int).to_numpy()
    ordered = train_df[[config.id_col, config.target_col]].copy()
    ordered["_position"] = np.arange(len(train_df))
    ordered = ordered.sort_values(config.id_col).reset_index(drop=True)
    ordered_positions = ordered["_position"].to_numpy().astype(int)

    best = None
    for lambda_transition in lambda_grid:
        predictions = np.full(len(train_df), fill_value=class_values[0], dtype=int)
        evaluated_mask = np.zeros(len(train_df), dtype=bool)
        for train_idx, valid_idx in fold_list:
            known_positions = set(np.asarray(train_idx).astype(int).tolist())
            hidden_positions = set(np.asarray(valid_idx).astype(int).tolist())
            known_df = train_df.iloc[list(known_positions)]
            transition = estimate_transition_matrix(
                known_df[config.target_col].astype(int),
                classes=class_values,
                ids=known_df[config.id_col],
                alpha=alpha_transition,
            )

            sequence_emissions = np.full((len(train_df), len(class_values)), 1.0 / len(class_values), dtype=float)
            allowed: list[set[int] | None] = []
            for position in ordered_positions:
                if position in hidden_positions:
                    sequence_emissions[len(allowed)] = emissions[position]
                    allowed.append(None)
                elif position in known_positions:
                    label = int(y_true[position])
                    allowed.append({class_to_idx[label]})
                else:
                    label = int(y_true[position])
                    allowed.append({class_to_idx[label]})

            decoded_indices = viterbi_decode(
                sequence_emissions,
                transition,
                allowed_label_indices=allowed,
                lambda_transition=lambda_transition,
            )
            decoded_labels_ordered = class_values[np.asarray(decoded_indices)].astype(int)
            for ordered_idx, position in enumerate(ordered_positions):
                if position in hidden_positions:
                    predictions[position] = decoded_labels_ordered[ordered_idx]
                    evaluated_mask[position] = True

        score = f1_score(y_true[evaluated_mask], predictions[evaluated_mask], average="macro", zero_division=0)
        candidate = {
            "lambda_transition": float(lambda_transition),
            "f1_macro": float(score),
            "predictions": predictions,
            "evaluated_mask": evaluated_mask,
        }
        if best is None or candidate["f1_macro"] > best["f1_macro"]:
            best = candidate
    return best


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
            "model_id": "rufimelo/Legal-BERTimbau-sts-base-ma-v2",
            "family": "Legal-BERTimbau Base",
            "why": "Portuguese legal-domain BERTimbau adaptation; safest first legal encoder on 12GB GPUs.",
            "max_length": 512,
        },
        {
            "model_id": "rufimelo/Legal-BERTimbau-sts-large-ma-v3",
            "family": "Legal-BERTimbau Large",
            "why": "Best-effort legal-domain encoder for stronger GPUs; tuned from Legal-BERTimbau large.",
            "max_length": 512,
        },
        {
            "model_id": "Tropic-AI/moBERTo",
            "family": "moBERTo",
            "why": "ModernBERT-style Portuguese encoder with long-context support; useful as a 2048/4096-token ablation.",
            "max_length": 8192,
        },
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
        "recommended_transformer": "rufimelo/Legal-BERTimbau-sts-base-ma-v2",
        "recommended_large_transformer": "rufimelo/Legal-BERTimbau-sts-large-ma-v3",
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


def build_label_mappings(labels: Sequence[int]) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """Build explicit label mappings for arbitrary integer class values."""

    label_values = sorted(pd.Series(labels).dropna().astype(int).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(label_values)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_values, label2id, id2label


def tokenize_head_tail_text(
    text: object,
    tokenizer,
    max_length: int = 512,
    head_tokens: int | None = None,
    tail_tokens: int | None = None,
) -> dict[str, list[int]]:
    """Tokenize one document with first-token plus last-token truncation."""

    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if (head_tokens is not None and head_tokens < 0) or (tail_tokens is not None and tail_tokens < 0):
        raise ValueError("head_tokens and tail_tokens must be non-negative.")

    encoded = tokenizer(str(text or ""), add_special_tokens=False, truncation=False)
    input_ids = list(encoded["input_ids"])
    try:
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    except AttributeError:
        special_count = 2
    content_length = max(max_length - special_count, 1)
    if len(input_ids) > content_length:
        if head_tokens is None and tail_tokens is None:
            tail_count = max(1, content_length // 4) if content_length > 1 else 0
            head_count = content_length - tail_count
        elif head_tokens is None:
            tail_count = min(int(tail_tokens or 0), content_length)
            head_count = content_length - tail_count
        elif tail_tokens is None:
            head_count = min(int(head_tokens), content_length)
            tail_count = content_length - head_count
        else:
            tail_count = min(int(tail_tokens), content_length)
            head_count = min(int(head_tokens), content_length - tail_count)
            remaining = content_length - head_count - tail_count
            head_count += remaining
        input_ids = input_ids[:head_count] + (input_ids[-tail_count:] if tail_count else [])

    if hasattr(tokenizer, "prepare_for_model"):
        return tokenizer.prepare_for_model(
            input_ids,
            truncation=False,
            max_length=max_length,
        )

    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        encoded_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    else:
        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        encoded_ids = input_ids
        if cls_id is not None:
            encoded_ids = [cls_id, *encoded_ids]
        if sep_id is not None:
            encoded_ids = [*encoded_ids, sep_id]

    result = {
        "input_ids": encoded_ids[:max_length],
        "attention_mask": [1] * min(len(encoded_ids), max_length),
    }
    if hasattr(tokenizer, "create_token_type_ids_from_sequences"):
        result["token_type_ids"] = tokenizer.create_token_type_ids_from_sequences(input_ids)[:max_length]
    return result


def tokenize_head_tail_batch(
    texts: Iterable[object],
    tokenizer,
    max_length: int = 512,
    head_tokens: int | None = None,
    tail_tokens: int | None = None,
) -> dict[str, list[list[int]]]:
    """Tokenize a batch using head+tail truncation without padding."""

    encoded_rows = [
        tokenize_head_tail_text(
            text,
            tokenizer,
            max_length=max_length,
            head_tokens=head_tokens,
            tail_tokens=tail_tokens,
        )
        for text in texts
    ]
    keys = sorted({key for row in encoded_rows for key in row})
    return {key: [row.get(key, []) for row in encoded_rows] for key in keys}


def fine_tune_transformer_classifier(
    train_df: pd.DataFrame,
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    config: ExperimentConfig | None = None,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    max_length: int = 512,
    output_dir: str = "outputs/transformer",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    gradient_checkpointing: bool = False,
    use_class_weights: bool = True,
    use_lora: bool = False,
    quantization_4bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    device: str | None = None,
    validation_size: float | None = None,
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
    label_values, label2id, id2label = build_label_mappings(data["label"])
    data["label"] = data["label"].map(label2id).astype(int)
    validation_fraction = config.validation_size if validation_size is None else float(validation_size)
    if validation_fraction > 0:
        train_split, valid_split = train_test_split(
            data,
            test_size=validation_fraction,
            random_state=config.random_state,
            stratify=data["label"],
        )
    else:
        train_split = data
        valid_split = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_kwargs = {
        "num_labels": len(label_values),
        "id2label": {idx: str(label) for idx, label in id2label.items()},
        "label2id": {str(label): idx for label, idx in label2id.items()},
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
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

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
        return tokenize_head_tail_batch(batch["text"], tokenizer, max_length=max_length)

    train_dataset = Dataset.from_pandas(train_split.reset_index(drop=True)).map(tokenize, batched=True)
    valid_dataset = (
        Dataset.from_pandas(valid_split.reset_index(drop=True)).map(tokenize, batched=True)
        if valid_split is not None
        else None
    )
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
            loss = loss_fct(logits.view(-1, _num_labels_from_logits(model, logits)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    class_weights = None
    if use_class_weights:
        counts = train_split["label"].value_counts().sort_index()
        weights = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor([weights.get(idx, 1.0) for idx in range(len(label_values))], dtype=torch.float)

    args_kwargs = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": max(1, int(gradient_accumulation_steps)),
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "gradient_checkpointing": gradient_checkpointing,
        "save_strategy": "epoch",
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": valid_dataset is not None,
        "report_to": "none",
        "fp16": bool(device == "cuda" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        "bf16": bool(device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    if valid_dataset is not None:
        args_kwargs["metric_for_best_model"] = "f1_macro"
        args_kwargs["greater_is_better"] = True
    # if device == "cpu":
    #     args_kwargs["use_cpu"] = True
    # elif device == "mps":
    #     args_kwargs["use_mps_device"] = True
    # try:
    #     args = TrainingArguments(evaluation_strategy="epoch", **args_kwargs)
    # except TypeError:
    #     try:
    #         args = TrainingArguments(eval_strategy="epoch", **args_kwargs)
    #     except TypeError:
    #         args_kwargs.pop("use_mps_device", None)
    #         args = TrainingArguments(eval_strategy="epoch", **args_kwargs)

    if device == "cpu":
        args_kwargs["use_cpu"] = True
    elif device == "mps":
        args_kwargs["use_mps_device"] = True

    # =========================================================================
    # SOLUÇÃO DEFINITIVA E ROBUSTA CONTRA MUDANÇAS DE VERSÃO DO TRANSFORMERS
    # =========================================================================
    # Filtra args_kwargs para conter APENAS parâmetros que a sua versão aceita
    import inspect
    valid_params = inspect.signature(TrainingArguments.__init__).parameters
    filtered_args = {k: v for k, v in args_kwargs.items() if k in valid_params}

    # Trata de forma inteligente a mudança de 'evaluation_strategy' para 'eval_strategy'
    eval_strategy_value = "epoch" if valid_dataset is not None else "no"
    if "eval_strategy" in valid_params:
        args = TrainingArguments(eval_strategy=eval_strategy_value, **filtered_args)
    else:
        args = TrainingArguments(evaluation_strategy=eval_strategy_value, **filtered_args)
    # =========================================================================

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    if valid_dataset is not None:
        trainer_kwargs["eval_dataset"] = valid_dataset
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


def load_transformer_trainer(
    model_dir: str | Path,
    device: str | None = None,
):
    """Load a saved HuggingFace sequence classifier as a Trainer."""

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer
    except ImportError as exc:
        raise ImportError("Install transformers and torch to load transformer trainers.") from exc

    import torch

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Transformer model directory not found: {model_dir}")
    device = device or gpu_environment_report().get("preferred_device", "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(torch.device(device))
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.processing_class = tokenizer
    return trainer


def _num_labels_from_logits(model, logits) -> int:
    """Return class count without assuming the model exposes config directly."""

    shape = getattr(logits, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    config = getattr(model, "config", None) or getattr(getattr(model, "module", None), "config", None)
    if config is not None and getattr(config, "num_labels", None) is not None:
        return int(config.num_labels)
    raise AttributeError("Could not infer num_labels from logits shape or model config.")


def _label_values_from_trainer(trainer, n_outputs: int) -> list[object]:
    """Recover original class labels from a HuggingFace Trainer."""

    config = getattr(getattr(trainer, "model", None), "config", None)
    id2label = getattr(config, "id2label", None) if config is not None else None
    if not id2label:
        return list(range(n_outputs))

    label_values: list[object] = []
    for idx in range(n_outputs):
        raw_label = id2label.get(idx, id2label.get(str(idx), idx))
        try:
            raw_label = int(raw_label)
        except (TypeError, ValueError):
            pass
        label_values.append(raw_label)
    return label_values


def predict_transformer_probabilities(
    trainer,
    test_df: pd.DataFrame,
    tokenizer=None,
    text_col: str = "Body_clean",
    max_length: int = 512,
) -> tuple[np.ndarray, list[object]]:
    """Return Transformer class probabilities and the matching original labels."""

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install datasets to generate transformer probabilities.") from exc

    if text_col not in test_df.columns:
        raise KeyError(f"Column '{text_col}' not found in test_df.")

    tokenizer = tokenizer or getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("A tokenizer or trainer.processing_class is required.")

    dataset = Dataset.from_pandas(test_df[[text_col]].rename(columns={text_col: "text"}).reset_index(drop=True))

    def tokenize(batch):
        return tokenize_head_tail_batch(batch["text"], tokenizer, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    removable_columns = [column for column in ("text", "__index_level_0__") if column in tokenized.column_names]
    if removable_columns:
        tokenized = tokenized.remove_columns(removable_columns)
    raw_predictions = trainer.predict(tokenized).predictions
    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]
    probabilities = _softmax(np.asarray(raw_predictions, dtype=float))
    return probabilities, _label_values_from_trainer(trainer, probabilities.shape[1])


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

    if text_col not in test_df.columns:
        raise KeyError(f"Column '{text_col}' not found in test_df.")
    if id_col not in test_df.columns:
        raise KeyError(f"Column '{id_col}' not found in test_df.")

    probabilities, label_values = predict_transformer_probabilities(
        trainer,
        test_df,
        tokenizer=tokenizer,
        text_col=text_col,
        max_length=max_length,
    )
    predicted_ids = np.argmax(probabilities, axis=-1).astype(int)
    labels = np.asarray(label_values, dtype=object)[predicted_ids]
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


def _predict_proba_aligned(model, texts: Iterable[str], classes: np.ndarray) -> np.ndarray:
    """Return probabilities with columns ordered exactly as ``classes``."""

    if hasattr(model, "predict_proba"):
        raw_scores = np.asarray(model.predict_proba(texts), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", classes)).astype(int)
        probabilities = raw_scores
    elif hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(texts), dtype=float)
        if raw_scores.ndim == 1:
            raw_scores = np.column_stack([-raw_scores, raw_scores])
        model_classes = np.asarray(getattr(model, "classes_", classes[: raw_scores.shape[1]])).astype(int)
        probabilities = _softmax(raw_scores)
    else:
        predictions = np.asarray(model.predict(texts)).astype(int)
        model_classes = classes.astype(int)
        probabilities = np.zeros((len(predictions), len(model_classes)), dtype=float)
        for row_idx, label in enumerate(predictions):
            matches = np.where(model_classes == label)[0]
            if len(matches):
                probabilities[row_idx, matches[0]] = 1.0

    aligned = np.zeros((probabilities.shape[0], len(classes)), dtype=float)
    class_to_col = {int(label): idx for idx, label in enumerate(classes.astype(int))}
    for source_idx, label in enumerate(model_classes):
        target_idx = class_to_col.get(int(label))
        if target_idx is not None and source_idx < probabilities.shape[1]:
            aligned[:, target_idx] = probabilities[:, source_idx]
    return _normalize_probabilities(aligned)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Normalize rows, replacing all-zero rows with a uniform distribution."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        total = values.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones_like(values, dtype=float) / len(values)
        return values / total
    if values.ndim != 2:
        raise ValueError("probabilities must be a 1D or 2D array.")
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    row_sums = values.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0
    if np.any(zero_rows):
        values[zero_rows] = 1.0 / values.shape[1]
        row_sums = values.sum(axis=1, keepdims=True)
    return values / np.maximum(row_sums, 1e-12)


def _allowed_mask(n_classes: int, allowed: set[int] | None) -> np.ndarray:
    """Create an additive log-space mask for allowed label indices."""

    mask = np.zeros(n_classes, dtype=float)
    if allowed is None:
        return mask
    mask[:] = -np.inf
    for idx in allowed:
        if idx < 0 or idx >= n_classes:
            raise ValueError(f"Allowed label index {idx} is outside 0..{n_classes - 1}.")
        mask[idx] = 0.0
    return mask


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
