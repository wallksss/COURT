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

"""Text preprocessing utilities for the legal document classifier.

The functions in this module are intentionally small and composable.  The
notebook can swap one cleaning/tokenization strategy for another while keeping
the experiment and analysis code unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "Acordao",
    1: "Agravo de Recurso Extraordinario (ARE)",
    2: "Despacho",
    3: "Recurso Extraordinario (RE)",
    4: "Sentenca",
}


DEFAULT_STOPWORDS_PT = {
    "a",
    "ao",
    "aos",
    "aquela",
    "aquele",
    "aqueles",
    "as",
    "ate",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "dele",
    "deles",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "era",
    "essa",
    "esse",
    "esta",
    "este",
    "foi",
    "ha",
    "isso",
    "ja",
    "lhe",
    "mais",
    "mas",
    "me",
    "mesmo",
    "na",
    "nas",
    "nem",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelo",
    "por",
    "que",
    "se",
    "sem",
    "ser",
    "seu",
    "sua",
    "tambem",
    "um",
    "uma",
}


LEGAL_STOPWORDS = {
    "autos",
    "brasilia",
    "certifico",
    "codigo",
    "colendo",
    "consta",
    "data",
    "decisao",
    "documento",
    "dou",
    "fls",
    "folha",
    "folhas",
    "julgamento",
    "ministro",
    "ministra",
    "processo",
    "relator",
    "relatora",
    "stf",
    "supremo",
    "tribunal",
}


LEGAL_KEYWORDS = {
    "kw_acordao": ("acordao", "colegiado", "turma", "plenario", "ementa"),
    "kw_are": ("agravo em recurso extraordinario", "agravo de recurso extraordinario", "are"),
    "kw_despacho": ("despacho", "intime", "publique", "vista", "certidao"),
    "kw_re": ("recurso extraordinario", "re ", " repercussao geral"),
    "kw_sentenca": ("sentenca", "julgo", "procedente", "improcedente", "extingo"),
    "kw_ocr_noise": ("assinatura", "carimbo", "digitalmente", "autenticacao", "ocr"),
}


@dataclass(frozen=True)
class TextPreprocessConfig:
    """Configuration for the deterministic text cleaner."""

    lowercase: bool = True
    strip_accents: bool = True
    normalize_legal_refs: bool = True
    remove_urls_emails: bool = True
    remove_stopwords: bool = False
    min_token_len: int = 2
    keep_digits: bool = False


def strip_accents(text: str) -> str:
    """Remove accents using Unicode decomposition."""

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces and line breaks."""

    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_ocr_artifacts(text: str) -> str:
    """Reduce common OCR artifacts without deleting useful legal content."""

    replacements = {
        "\x0c": " ",
        "\u00a0": " ",
        "§": " artigo ",
        "º": " ",
        "ª": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"([a-zA-Z])\s*-\s+([a-zA-Z])", r"\1\2", text)
    text = re.sub(r"[_=]{2,}", " ", text)
    text = re.sub(r"\.{3,}", " ... ", text)
    return normalize_whitespace(text)


def normalize_legal_references(text: str) -> str:
    """Normalize process numbers, dates, money values and article references."""

    text = re.sub(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b", " NUM_PROCESSO ", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " DATA ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " DATA ", text)
    text = re.sub(r"R\$\s*[\d\.\,]+", " VALOR_MONETARIO ", text)
    text = re.sub(r"\bart\.?\s*\d+[a-zA-Z]?\b", " ARTIGO_LEI ", text, flags=re.IGNORECASE)
    text = re.sub(r"\blei\s*n[ºo]?\s*[\d\.\-/]+", " LEI ", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)


def tokenize_simple(text: str, min_token_len: int = 2, keep_digits: bool = False) -> list[str]:
    """Tokenize with a regex that is robust to noisy OCR text."""

    if keep_digits:
        tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", text)
    else:
        tokens = re.findall(r"[A-Za-zÀ-ÿ_]+", text)
    return [token for token in tokens if len(token) >= min_token_len]


def remove_stopwords(tokens: Sequence[str], extra_stopwords: Iterable[str] | None = None) -> list[str]:
    """Remove Portuguese and legal-domain stopwords."""

    stopwords = set(DEFAULT_STOPWORDS_PT) | set(LEGAL_STOPWORDS)
    if extra_stopwords:
        stopwords.update(extra_stopwords)
    return [token for token in tokens if token not in stopwords]


def clean_text(text: object, config: TextPreprocessConfig | None = None) -> str:
    """Apply the deterministic preprocessing strategy to one text."""

    config = config or TextPreprocessConfig()
    if pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = fix_ocr_artifacts(cleaned)

    if config.remove_urls_emails:
        cleaned = re.sub(r"https?://\S+|www\.\S+", " URL ", cleaned)
        cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " EMAIL ", cleaned)

    if config.normalize_legal_refs:
        cleaned = normalize_legal_references(cleaned)

    if config.lowercase:
        cleaned = cleaned.lower()

    if config.strip_accents:
        cleaned = strip_accents(cleaned)

    tokens = tokenize_simple(
        cleaned,
        min_token_len=config.min_token_len,
        keep_digits=config.keep_digits,
    )

    if config.remove_stopwords:
        tokens = remove_stopwords(tokens)

    return " ".join(tokens)


def preprocess_corpus(
    texts: Sequence[object],
    config: TextPreprocessConfig | None = None,
) -> pd.Series:
    """Clean a sequence of documents and return a pandas Series."""

    return pd.Series(texts, dtype="object").map(lambda value: clean_text(value, config=config))


def add_clean_text_column(
    df: pd.DataFrame,
    text_col: str = "Body",
    output_col: str = "Body_clean",
    config: TextPreprocessConfig | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with an additional cleaned text column."""

    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found.")
    result = df.copy()
    result[output_col] = preprocess_corpus(result[text_col], config=config).values
    return result


def make_sklearn_preprocessor(config: TextPreprocessConfig | None = None) -> Callable[[object], str]:
    """Create a callable compatible with scikit-learn vectorizers."""

    return lambda value: clean_text(value, config=config)


def chunk_text(text: object, max_words: int = 350, overlap_words: int = 60) -> list[str]:
    """Split a long document into overlapping word chunks."""

    words = str(text if not pd.isna(text) else "").split()
    if not words:
        return [""]
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words.")

    chunks = []
    start = 0
    step = max_words - overlap_words
    while start < len(words):
        chunks.append(" ".join(words[start : start + max_words]))
        start += step
    return chunks


def extract_legal_signal_features(
    df: pd.DataFrame,
    text_col: str = "Body_clean",
) -> pd.DataFrame:
    """Extract lightweight domain features from legal texts.

    These features complement sparse/dense text representations and are useful
    for error analysis.  They do not replace TF-IDF or transformer embeddings.
    """

    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found.")

    rows = []
    for text in df[text_col].fillna("").astype(str):
        word_count = max(len(text.split()), 1)
        row = {
            "feat_word_count": word_count,
            "feat_char_count": len(text),
            "feat_digit_ratio": _safe_ratio(sum(ch.isdigit() for ch in text), len(text)),
            "feat_upper_ratio": _safe_ratio(sum(ch.isupper() for ch in text), len(text)),
            "feat_unique_word_ratio": _safe_ratio(len(set(text.split())), word_count),
            "feat_process_ref_count": text.count("num_processo"),
            "feat_article_ref_count": text.count("artigo_lei"),
            "feat_date_ref_count": text.count("data"),
        }
        for feature_name, keywords in LEGAL_KEYWORDS.items():
            row[feature_name] = sum(text.count(keyword) for keyword in keywords) / word_count
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def extract_basic_nlp_features(
    df: pd.DataFrame,
    text_col: str = "Body",
    spacy_model: str = "pt_core_news_sm",
    max_docs: int | None = None,
) -> pd.DataFrame:
    """Extract optional PoS/NER counts with spaCy, falling back gracefully.

    The project PDF asks for basic NLP tasks where appropriate.  This function
    uses spaCy when the Portuguese model is installed. If not, it returns the
    legal signal features computed by ``extract_legal_signal_features``.
    """

    try:
        import spacy
    except ImportError:
        temp = add_clean_text_column(df, text_col=text_col)
        return extract_legal_signal_features(temp, text_col="Body_clean")

    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        temp = add_clean_text_column(df, text_col=text_col)
        return extract_legal_signal_features(temp, text_col="Body_clean")

    rows = []
    texts = df[text_col].fillna("").astype(str)
    if max_docs is not None:
        texts = texts.head(max_docs)

    for doc in nlp.pipe(texts, batch_size=16):
        token_count = max(len(doc), 1)
        pos_counts = {}
        for token in doc:
            pos_counts[f"pos_{token.pos_.lower()}"] = pos_counts.get(f"pos_{token.pos_.lower()}", 0) + 1
        ent_counts = {}
        for ent in doc.ents:
            ent_counts[f"ner_{ent.label_.lower()}"] = ent_counts.get(f"ner_{ent.label_.lower()}", 0) + 1

        row = {key: value / token_count for key, value in pos_counts.items()}
        row.update({key: value / token_count for key, value in ent_counts.items()})
        row["nlp_token_count"] = token_count
        row["nlp_entity_count"] = len(doc.ents)
        rows.append(row)

    features = pd.DataFrame(rows, index=texts.index).fillna(0.0)
    if len(features) < len(df):
        features = features.reindex(df.index, fill_value=0.0)
    return features


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
