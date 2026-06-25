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

"""Notebook orchestration helpers with lightweight file caching."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .analise_exploratoria import load_competition_data, split_labeled_unlabeled, validate_schema
from .preprocessamento import TextPreprocessConfig, add_clean_text_column


def resolve_project_dir(start: str | Path | None = None) -> Path:
    """Return the notebook project directory containing ``scripts/``."""

    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "scripts").is_dir() and (candidate / "data").is_dir():
            return candidate
        nested = candidate / "template-implementacao"
        if (nested / "scripts").is_dir() and (nested / "data").is_dir():
            return nested
    return current


def cached_preprocessed_data(
    project_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config: TextPreprocessConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Load raw CSVs and reuse preprocessed CSVs when available."""

    project_path = resolve_project_dir(project_dir)
    data_path = Path(data_dir) if data_dir is not None else project_path / "data"
    output_path = Path(output_dir) if output_dir is not None else project_path / "outputs"
    if not data_path.is_absolute():
        data_path = project_path / data_path
    if not output_path.is_absolute():
        output_path = project_path / output_path

    cache_dir = output_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = config or TextPreprocessConfig()
    train_cache = cache_dir / "train_clean.csv"
    test_cache = cache_dir / "test_clean.csv"
    unlabeled_cache = cache_dir / "train_unlabeled_clean.csv"
    meta_cache = cache_dir / "preprocess_metadata.json"

    train_df, test_df, sample_df = load_competition_data(data_path)
    schema_report = validate_schema(train_df, test_df, sample_df)
    train_labeled, train_unlabeled = split_labeled_unlabeled(train_df)

    expected_meta = {
        "config": asdict(config),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "labeled_rows": int(len(train_labeled)),
        "unlabeled_rows": int(len(train_unlabeled)),
    }
    cache_ready = train_cache.exists() and test_cache.exists() and unlabeled_cache.exists()
    if cache_ready and meta_cache.exists():
        try:
            cache_ready = json.loads(meta_cache.read_text(encoding="utf-8")) == expected_meta
        except json.JSONDecodeError:
            cache_ready = False

    if force or not cache_ready:
        train_clean = add_clean_text_column(train_labeled, config=config)
        test_clean = add_clean_text_column(test_df, config=config)
        train_unlabeled_clean = add_clean_text_column(train_unlabeled, config=config)
        train_clean.to_csv(train_cache, index=False)
        test_clean.to_csv(test_cache, index=False)
        train_unlabeled_clean.to_csv(unlabeled_cache, index=False)
        meta_cache.write_text(json.dumps(expected_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        cache_status = "rebuilt"
    else:
        train_clean = pd.read_csv(train_cache)
        test_clean = pd.read_csv(test_cache)
        train_unlabeled_clean = pd.read_csv(unlabeled_cache)
        cache_status = "loaded"

    return {
        "project_dir": project_path,
        "data_dir": data_path,
        "output_dir": output_path,
        "cache_dir": cache_dir,
        "train_df": train_df,
        "test_df": test_df,
        "sample_df": sample_df,
        "schema_report": schema_report,
        "train_labeled": train_labeled,
        "train_unlabeled": train_unlabeled,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "train_unlabeled_clean": train_unlabeled_clean,
        "cache_status": cache_status,
    }


def read_cached_csv(path: str | Path) -> pd.DataFrame | None:
    """Read a cached CSV if it exists; otherwise return ``None``."""

    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_cached_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame cache and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
