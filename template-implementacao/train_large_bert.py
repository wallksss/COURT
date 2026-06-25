#!/usr/bin/env python3
"""Final one-shot Legal-BERTimbau Large training script.

This entry point intentionally does not depend on the notebook's experiment
helpers. It rebuilds preprocessing from ``data/train.csv`` and ``data/test.csv``,
fine-tunes a large BERT model on all valid labeled rows, and writes one final
submission CSV.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.preprocessamento import TextPreprocessConfig, add_clean_text_column  # noqa: E402


DEFAULT_MODEL_NAME = "rufimelo/Legal-BERTimbau-sts-large-ma-v3"
VALID_LABELS = (0, 1, 2, 3, 4)


def build_preprocess_config() -> TextPreprocessConfig:
    """Return the deterministic preprocessing used for the final experiment."""

    return TextPreprocessConfig(
        lowercase=True,
        strip_accents=True,
        normalize_legal_refs=True,
        remove_urls_emails=True,
        remove_stopwords=False,
        min_token_len=2,
        keep_digits=False,
    )


def default_hyperparameters_for_device(device: str) -> dict[str, float | int]:
    """Choose practical defaults for BERT large on the selected accelerator."""

    if device == "cuda":
        batch_size = 4
        gradient_accumulation_steps = 4
    else:
        batch_size = 1
        gradient_accumulation_steps = 1
    return {
        "epochs": 4.0,
        "learning_rate": 1e-5,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "max_length": 512,
    }


def load_and_preprocess_data(
    data_dir: str | Path,
    preprocess_config: TextPreprocessConfig,
    valid_labels: Sequence[int] = VALID_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load raw competition CSVs and rebuild clean text columns from scratch."""

    data_path = Path(data_dir).resolve()
    train_path = data_path / "train.csv"
    test_path = data_path / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv inside {data_path}. "
            "This script does not reuse preprocessing cache by default."
        )

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    _validate_input_columns(train_raw, required=("Id", "Body", "Category"), frame_name="train.csv")
    _validate_input_columns(test_raw, required=("Id", "Body"), frame_name="test.csv")

    valid_label_set = set(int(label) for label in valid_labels)
    labeled = train_raw.loc[train_raw["Category"].isin(valid_label_set)].copy().reset_index(drop=True)
    labeled["Category"] = labeled["Category"].astype(int)
    if labeled.empty:
        raise ValueError("No valid labeled rows found in train.csv.")

    train_clean = add_clean_text_column(
        labeled,
        text_col="Body",
        output_col="Body_clean",
        config=preprocess_config,
    )
    test_clean = add_clean_text_column(
        test_raw.copy(),
        text_col="Body",
        output_col="Body_clean",
        config=preprocess_config,
    )
    report = {
        "raw_train_rows": int(len(train_raw)),
        "final_train_rows": int(len(train_clean)),
        "dropped_unlabeled_rows": int(len(train_raw) - len(train_clean)),
        "test_rows": int(len(test_clean)),
    }
    return train_clean, test_clean, report


def build_label_mappings(labels: Iterable[int]) -> tuple[list[int], dict[int, int], dict[int, int]]:
    label_values = sorted(pd.Series(list(labels)).dropna().astype(int).unique().tolist())
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
    """Tokenize with head+tail truncation for long legal documents."""

    encoded = tokenizer(str(text or ""), add_special_tokens=False, truncation=False)
    input_ids = list(encoded["input_ids"])
    try:
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    except AttributeError:
        special_count = 2
    content_length = max(int(max_length) - special_count, 1)

    if len(input_ids) > content_length:
        if head_tokens is None and tail_tokens is None:
            tail_count = int(np.ceil(content_length / 4)) if content_length > 1 else 0
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
            head_count += content_length - head_count - tail_count
        input_ids = input_ids[:head_count] + (input_ids[-tail_count:] if tail_count else [])

    if hasattr(tokenizer, "prepare_for_model"):
        return tokenizer.prepare_for_model(input_ids, truncation=False, max_length=max_length)

    encoded_ids = input_ids
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        encoded_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    else:
        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        if cls_id is not None:
            encoded_ids = [cls_id, *encoded_ids]
        if sep_id is not None:
            encoded_ids = [*encoded_ids, sep_id]
    return {"input_ids": encoded_ids[:max_length], "attention_mask": [1] * min(len(encoded_ids), max_length)}


def tokenize_head_tail_batch(
    texts: Iterable[object],
    tokenizer,
    max_length: int = 512,
) -> dict[str, list[list[int]]]:
    rows = [tokenize_head_tail_text(text, tokenizer, max_length=max_length) for text in texts]
    keys = sorted({key for row in rows for key in row})
    return {key: [row.get(key, []) for row in rows] for key in keys}


def train_final_model(
    train_df: pd.DataFrame,
    model_name: str,
    output_model_dir: str | Path,
    device: str,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    use_class_weights: bool = True,
    gradient_checkpointing: bool = True,
):
    """Fine-tune the sequence classifier on all preprocessed labeled rows."""

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Install torch, transformers, datasets and accelerate before running this script."
        ) from exc

    set_global_seed(seed)
    label_values, label2id, id2label = build_label_mappings(train_df["Category"])
    training = train_df[["Body_clean", "Category"]].rename(columns={"Body_clean": "text", "Category": "label"})
    training["label"] = training["label"].map(label2id).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_values),
        id2label={idx: str(label) for idx, label in id2label.items()},
        label2id={str(label): idx for label, idx in label2id.items()},
    )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.to(torch.device(device))

    dataset = Dataset.from_pandas(training.reset_index(drop=True)).map(
        lambda batch: tokenize_head_tail_batch(batch["text"], tokenizer, max_length=max_length),
        batched=True,
    )
    dataset = dataset.remove_columns([column for column in ["text", "__index_level_0__"] if column in dataset.column_names])

    class_weights = None
    if use_class_weights:
        counts = training["label"].value_counts().sort_index()
        weights = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor([weights.get(idx, 1.0) for idx in range(len(label_values))], dtype=torch.float)

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
            loss_fct = (
                torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                if self.class_weights is not None
                else torch.nn.CrossEntropyLoss()
            )
            loss = loss_fct(logits.view(-1, num_labels_from_logits(model, logits)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args = make_training_arguments(
        TrainingArguments,
        output_dir=output_model_dir,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_checkpointing=gradient_checkpointing,
    )
    trainer_cls = WeightedTrainer if use_class_weights else Trainer
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    }
    if use_class_weights:
        trainer_kwargs["class_weights"] = class_weights
    trainer = instantiate_trainer(trainer_cls, tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    return trainer


def make_training_arguments(
    training_arguments_cls,
    output_dir: str | Path,
    device: str,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    weight_decay: float,
    warmup_ratio: float,
    gradient_checkpointing: bool,
):
    """Create TrainingArguments across transformers versions."""

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "per_device_train_batch_size": int(batch_size),
        "per_device_eval_batch_size": int(batch_size),
        "gradient_accumulation_steps": max(1, int(gradient_accumulation_steps)),
        "num_train_epochs": float(epochs),
        "weight_decay": float(weight_decay),
        "warmup_ratio": float(warmup_ratio),
        "lr_scheduler_type": "linear",
        "gradient_checkpointing": bool(gradient_checkpointing),
        "save_strategy": "epoch",
        "logging_steps": 50,
        "save_total_limit": 1,
        "report_to": "none",
    }
    try:
        import torch

        kwargs["fp16"] = bool(device == "cuda" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
        kwargs["bf16"] = bool(device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        kwargs["fp16"] = False
        kwargs["bf16"] = False

    valid_params = inspect.signature(training_arguments_cls.__init__).parameters
    if device == "cpu" and "use_cpu" in valid_params:
        kwargs["use_cpu"] = True
    if device == "mps" and "use_mps_device" in valid_params:
        kwargs["use_mps_device"] = True
    filtered = {key: value for key, value in kwargs.items() if key in valid_params}
    if "eval_strategy" in valid_params:
        return training_arguments_cls(eval_strategy="no", **filtered)
    return training_arguments_cls(evaluation_strategy="no", **filtered)


def instantiate_trainer(trainer_cls, tokenizer, **kwargs):
    """Instantiate Trainer without relying on deprecated tokenizer kwargs."""

    try:
        supports_processing_class = "processing_class" in inspect.signature(trainer_cls.__init__).parameters
    except (TypeError, ValueError):
        supports_processing_class = False
    if supports_processing_class:
        kwargs["processing_class"] = tokenizer
    trainer = trainer_cls(**kwargs)
    if not hasattr(trainer, "processing_class") or getattr(trainer, "processing_class", None) is None:
        trainer.processing_class = tokenizer
    return trainer


def predict_submission(
    trainer,
    test_df: pd.DataFrame,
    output_path: str | Path,
    max_length: int,
) -> pd.DataFrame:
    """Predict test labels and write a single submission CSV."""

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install datasets to generate predictions.") from exc

    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Trainer has no tokenizer/processing_class attached.")

    dataset = Dataset.from_pandas(test_df[["Body_clean"]].rename(columns={"Body_clean": "text"}).reset_index(drop=True))
    dataset = dataset.map(lambda batch: tokenize_head_tail_batch(batch["text"], tokenizer, max_length=max_length), batched=True)
    dataset = dataset.remove_columns([column for column in ["text", "__index_level_0__"] if column in dataset.column_names])

    predictions = trainer.predict(dataset).predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    probs = softmax(np.asarray(predictions, dtype=float))
    class_labels = label_values_from_trainer(trainer, probs.shape[1])
    predicted = np.asarray(class_labels, dtype=int)[np.argmax(probs, axis=1)]
    submission = pd.DataFrame({"Id": test_df["Id"].values, "Category": predicted.astype(int)})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def label_values_from_trainer(trainer, n_outputs: int) -> list[int]:
    config = getattr(getattr(trainer, "model", None), "config", None)
    id2label = getattr(config, "id2label", None) if config is not None else None
    if not id2label:
        return list(range(n_outputs))
    labels = []
    for idx in range(n_outputs):
        raw_label = id2label.get(idx, id2label.get(str(idx), idx))
        labels.append(int(raw_label))
    return labels


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Legal-BERTimbau Large once, from raw CSV preprocessing through final submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-output-name", default="large_bert_final_full_train")
    parser.add_argument("--submission-name", default="submission_large_bert.csv")
    parser.add_argument("--metadata-name", default="large_bert_run_metadata.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--save-preprocessed", action="store_true")
    parser.set_defaults(use_class_weights=True, gradient_checkpointing=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    device = resolve_device(args.device)
    defaults = default_hyperparameters_for_device(device)
    epochs = float(args.epochs if args.epochs is not None else defaults["epochs"])
    learning_rate = float(args.learning_rate if args.learning_rate is not None else defaults["learning_rate"])
    batch_size = int(args.batch_size if args.batch_size is not None else defaults["batch_size"])
    gradient_accumulation_steps = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else defaults["gradient_accumulation_steps"]
    )
    max_length = int(args.max_length if args.max_length is not None else defaults["max_length"])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else defaults["weight_decay"])
    warmup_ratio = float(args.warmup_ratio if args.warmup_ratio is not None else defaults["warmup_ratio"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = args.model_dir / args.model_output_name
    submission_path = args.output_dir / args.submission_name
    metadata_path = args.output_dir / args.metadata_name

    preprocess_config = build_preprocess_config()
    train_df, test_df, data_report = load_and_preprocess_data(args.data_dir, preprocess_config)
    if args.save_preprocessed:
        train_df.to_csv(args.output_dir / "train_large_bert_preprocessed.csv", index=False)
        test_df.to_csv(args.output_dir / "test_large_bert_preprocessed.csv", index=False)

    print("Modelo:", args.model_name)
    print("Dispositivo:", device)
    print("Treino:", len(train_df), "linhas rotuladas")
    print("Teste:", len(test_df), "linhas")
    print("Validacao: desativada; treino usa todos os rotulos validos")
    print("Epocas:", epochs)
    print("Learning rate:", learning_rate)
    print("Max length:", max_length, "com truncamento head+tail")
    print("Batch efetivo:", batch_size * gradient_accumulation_steps)

    trainer = train_final_model(
        train_df=train_df,
        model_name=args.model_name,
        output_model_dir=model_output_dir,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    submission = predict_submission(trainer, test_df, submission_path, max_length=max_length)

    metadata = {
        "model_name": args.model_name,
        "model_output_dir": str(model_output_dir),
        "submission_path": str(submission_path),
        "final_training_uses_validation": False,
        "data_report": data_report,
        "preprocess_config": asdict(preprocess_config),
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "max_length": max_length,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "seed": args.seed,
            "device": device,
            "use_class_weights": args.use_class_weights,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "output_rows": int(len(submission)),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Submissao salva em:", submission_path)
    print("Metadados salvos em:", metadata_path)
    return 0


def _validate_input_columns(df: pd.DataFrame, required: Sequence[str], frame_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def num_labels_from_logits(model, logits) -> int:
    shape = getattr(logits, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    config = getattr(model, "config", None) or getattr(getattr(model, "module", None), "config", None)
    if config is not None and getattr(config, "num_labels", None) is not None:
        return int(config.num_labels)
    raise AttributeError("Could not infer num_labels from logits shape or model config.")


if __name__ == "__main__":
    raise SystemExit(main())
