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

"""Result analysis utilities for classification experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def rank_results(results_df: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Return a clean ranking table sorted by the requested metric."""

    if metric not in results_df.columns:
        raise KeyError(f"Metric '{metric}' not found in results_df.")
    cols = [col for col in ["model", metric, "f1_weighted", "accuracy", "precision_macro", "recall_macro"] if col in results_df]
    return results_df[cols].sort_values(metric, ascending=False).reset_index(drop=True)


def classification_report_frame(report: dict) -> pd.DataFrame:
    """Convert sklearn classification_report(output_dict=True) to DataFrame."""

    return pd.DataFrame(report).transpose()


def plot_metric_comparison(results_df: pd.DataFrame, metric: str = "f1_macro"):
    """Plot a horizontal model ranking."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    ranked = rank_results(results_df, metric=metric)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(ranked))))
    sns.barplot(data=ranked, x=metric, y="model", ax=ax, color="#2563eb")
    ax.set_title(f"Comparacao de modelos por {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("")
    ax.set_xlim(0, 1.0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(y_true, y_pred, labels: list[int] | None = None, normalize: bool = False):
    """Plot a confusion matrix for one validation run."""

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    labels = labels or sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao" + (" normalizada" if normalize else ""))
    fig.tight_layout()
    return fig, ax


def analyze_errors(
    validation_texts: pd.Series,
    y_true,
    y_pred,
    ids: pd.Series | None = None,
    max_chars: int = 600,
) -> pd.DataFrame:
    """Return a DataFrame with misclassified validation examples."""

    frame = pd.DataFrame(
        {
            "Id": ids.values if ids is not None else np.arange(len(validation_texts)),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "text": validation_texts.astype(str).values,
        }
    )
    errors = frame.loc[frame["y_true"] != frame["y_pred"]].copy()
    errors["text_preview"] = errors["text"].str.slice(0, max_chars)
    return errors.drop(columns=["text"]).reset_index(drop=True)


def error_profile_by_length(
    validation_df: pd.DataFrame,
    y_true,
    y_pred,
    length_col: str = "n_words",
) -> pd.DataFrame:
    """Compare error rate across document-length buckets."""

    if length_col not in validation_df.columns:
        raise KeyError(f"Column '{length_col}' not found.")
    frame = validation_df[[length_col]].copy()
    frame["is_error"] = np.asarray(y_true) != np.asarray(y_pred)
    frame["length_bucket"] = pd.qcut(frame[length_col].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    return (
        frame.groupby("length_bucket", observed=True)
        .agg(n_docs=("is_error", "size"), error_rate=("is_error", "mean"), avg_length=(length_col, "mean"))
        .reset_index()
    )


def save_results_table(results_df: pd.DataFrame, output_path: str | Path = "outputs/model_results.csv") -> Path:
    """Save model metrics to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return output_path


def compare_with_cv(holdout_df: pd.DataFrame, cv_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge holdout and cross-validation summaries when both are available."""

    if cv_df is None or cv_df.empty:
        return holdout_df.copy()
    return holdout_df.merge(cv_df, on="model", how="left", suffixes=("_holdout", "_cv"))
