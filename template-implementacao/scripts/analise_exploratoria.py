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

"""Exploratory analysis helpers for the Kaggle legal NLP competition."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


REQUIRED_TRAIN_COLUMNS = {"Id", "Body", "Category"}
REQUIRED_TEST_COLUMNS = {"Id", "Body"}
REQUIRED_SUBMISSION_COLUMNS = {"Id", "Category"}


def load_competition_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load train, test and sample submission CSV files.

    Expected files:
    - data/train.csv with Id, Body, Category
    - data/test.csv with Id, Body
    - data/sample_submission.csv with Id, Category
    """

    data_path = Path(data_dir)
    data_path = data_path if data_path.is_absolute() else Path.cwd() / data_path
    data_path = _resolve_data_dir(data_path)
    train_path = data_path / "train.csv"
    test_path = data_path / "test.csv"
    sample_path = data_path / "sample_submission.csv"

    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Competition CSV files were not found. Put train.csv, test.csv and "
            f"sample_submission.csv inside '{data_path.resolve()}'. Missing: {missing}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_path) if sample_path.exists() else None
    return train_df, test_df, sample_df


def _resolve_data_dir(data_path: Path) -> Path:
    """Find the directory containing the competition CSV files."""

    candidate_dirs = [
        data_path,
        data_path.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidate_dirs:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    return data_path


def validate_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Validate competition columns and return a compact integrity report."""

    report = {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "sample_shape": None if sample_df is None else sample_df.shape,
        "train_missing_columns": sorted(REQUIRED_TRAIN_COLUMNS - set(train_df.columns)),
        "test_missing_columns": sorted(REQUIRED_TEST_COLUMNS - set(test_df.columns)),
        "sample_missing_columns": None
        if sample_df is None
        else sorted(REQUIRED_SUBMISSION_COLUMNS - set(sample_df.columns)),
        "train_nulls": train_df[["Id", "Body", "Category"]].isna().sum().to_dict()
        if REQUIRED_TRAIN_COLUMNS.issubset(train_df.columns)
        else {},
        "test_nulls": test_df[["Id", "Body"]].isna().sum().to_dict()
        if REQUIRED_TEST_COLUMNS.issubset(test_df.columns)
        else {},
        "duplicated_train_ids": int(train_df["Id"].duplicated().sum()) if "Id" in train_df else None,
        "duplicated_test_ids": int(test_df["Id"].duplicated().sum()) if "Id" in test_df else None,
        "invalid_category_count": 0,
        "invalid_category_examples": [],
    }

    if "Category" in train_df.columns:
        invalid = train_df.loc[~train_df["Category"].between(0, 4), ["Id", "Category"]]
        report["invalid_category_count"] = int(len(invalid))
        report["invalid_category_examples"] = invalid.head(10).to_dict("records")

    missing_any = (
        report["train_missing_columns"]
        or report["test_missing_columns"]
        or (sample_df is not None and report["sample_missing_columns"])
    )
    if missing_any:
        raise ValueError(f"Invalid competition schema: {report}")
    return report


def split_labeled_unlabeled(
    train_df: pd.DataFrame,
    target_col: str = "Category",
    valid_labels: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows with valid labels from unlabeled/invalid rows."""

    if target_col not in train_df.columns:
        raise KeyError(f"Column '{target_col}' not found.")
    mask = train_df[target_col].isin(valid_labels)
    labeled = train_df.loc[mask].copy().reset_index(drop=True)
    unlabeled = train_df.loc[~mask].copy().reset_index(drop=True)
    labeled[target_col] = labeled[target_col].astype(int)
    return labeled, unlabeled


def add_text_statistics(df: pd.DataFrame, text_col: str = "Body") -> pd.DataFrame:
    """Return a copy with text length and noise statistics."""

    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found.")

    result = df.copy()
    text = result[text_col].fillna("").astype(str)
    words = text.str.split()

    result["n_chars"] = text.str.len()
    result["n_words"] = words.str.len()
    result["n_lines"] = text.str.count(r"\n") + 1
    result["n_unique_words"] = words.map(lambda value: len(set(value)))
    result["unique_word_ratio"] = np.where(
        result["n_words"] > 0,
        result["n_unique_words"] / result["n_words"],
        0.0,
    )
    result["avg_word_len"] = words.map(_average_word_length)
    result["digit_ratio"] = text.map(lambda value: _char_ratio(value, str.isdigit))
    result["uppercase_ratio"] = text.map(lambda value: _char_ratio(value, str.isupper))
    result["punct_ratio"] = text.map(lambda value: _char_ratio(value, lambda c: not c.isalnum() and not c.isspace()))
    return result


def summarize_dataset(train_df: pd.DataFrame, text_col: str = "Body", target_col: str = "Category") -> dict[str, pd.DataFrame]:
    """Build reusable summary tables for the training data."""

    train_stats = add_text_statistics(train_df, text_col=text_col)
    class_distribution = (
        train_stats[target_col]
        .value_counts(dropna=False)
        .rename_axis(target_col)
        .reset_index(name="count")
        .sort_values(target_col)
    )
    class_distribution["percent"] = class_distribution["count"] / len(train_stats) * 100

    length_summary = train_stats.groupby(target_col)[["n_chars", "n_words", "unique_word_ratio"]].describe()
    missing_summary = train_stats[["Id", text_col, target_col]].isna().sum().rename("missing").to_frame()

    return {
        "data_with_stats": train_stats,
        "class_distribution": class_distribution,
        "length_summary": length_summary,
        "missing_summary": missing_summary,
    }


def get_top_terms(
    df: pd.DataFrame,
    text_col: str = "Body",
    target_col: str = "Category",
    n_terms: int = 15,
    preprocessor: Callable[[object], str] | None = None,
    max_features: int = 15000,
) -> pd.DataFrame:
    """Return top TF-IDF terms per class."""

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError("Install scikit-learn to compute top terms.") from exc

    rows = []
    for category in sorted(df[target_col].dropna().unique()):
        texts = df.loc[df[target_col] == category, text_col].fillna("").astype(str)
        if texts.empty:
            continue

        vectorizer = TfidfVectorizer(
            preprocessor=preprocessor,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        features = np.array(vectorizer.get_feature_names_out())
        top_idx = scores.argsort()[::-1][:n_terms]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "Category": category,
                    "rank": rank,
                    "term": features[idx],
                    "mean_tfidf": scores[idx],
                }
            )
    return pd.DataFrame(rows)


def sample_texts_by_class(
    df: pd.DataFrame,
    target_col: str = "Category",
    n_per_class: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample examples from each class for qualitative analysis."""

    return (
        df.groupby(target_col, group_keys=False)
        .apply(lambda group: group.sample(min(n_per_class, len(group)), random_state=random_state))
        .reset_index(drop=True)
    )


def plot_class_distribution(class_distribution: pd.DataFrame, target_col: str = "Category"):
    """Plot class counts and percentages."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=class_distribution, x=target_col, y="count", ax=ax, color="#3b82f6")
    ax.set_title("Distribuicao das categorias")
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Quantidade")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    return fig, ax


def plot_text_length_distribution(
    df_stats: pd.DataFrame,
    target_col: str = "Category",
    metric: str = "n_words",
):
    """Plot histogram and boxplot of a text length metric."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(data=df_stats, x=metric, hue=target_col, bins=50, element="step", ax=axes[0])
    axes[0].set_title(f"Distribuicao de {metric}")
    axes[0].set_xlabel(metric)
    sns.boxplot(data=df_stats, x=target_col, y=metric, ax=axes[1], color="#93c5fd")
    axes[1].set_title(f"{metric} por categoria")
    axes[1].set_xlabel("Categoria")
    fig.tight_layout()
    return fig, axes


def plot_top_terms(top_terms: pd.DataFrame, n_terms: int = 12):
    """Plot the most relevant TF-IDF terms for each class."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    if top_terms.empty:
        raise ValueError("top_terms is empty.")
    data = top_terms[top_terms["rank"] <= n_terms].copy()
    grid = sns.catplot(
        data=data,
        y="term",
        x="mean_tfidf",
        col="Category",
        kind="bar",
        sharey=False,
        col_wrap=3,
        height=4,
        color="#2563eb",
    )
    grid.set_axis_labels("TF-IDF medio", "")
    grid.set_titles("Categoria {col_name}")
    plt.tight_layout()
    return grid


def _average_word_length(words: list[str]) -> float:
    if not words:
        return 0.0
    return float(np.mean([len(word) for word in words]))


def _char_ratio(text: str, predicate) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if predicate(char)) / len(text)
