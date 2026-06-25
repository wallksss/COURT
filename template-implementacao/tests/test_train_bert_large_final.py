import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_bert_large_final as trainer_script  # noqa: E402


class FinalBertScriptTests(unittest.TestCase):
    def _make_config(self, project: Path) -> trainer_script.FinalTrainingConfig:
        return trainer_script.FinalTrainingConfig(
            project_dir=project,
            data_dir=project / "data",
            output_dir=project / "outputs",
            model_dir=project / "models",
            model_name="base-model",
            validation_model_dir=project / "models" / "validation",
            final_model_dir=project / "models" / "final",
            raw_submission_path=project / "outputs" / "raw.csv",
            final_submission_path=project / "outputs" / "final.csv",
            metadata_path=project / "outputs" / "metadata.json",
            validation_report_path=project / "outputs" / "validation.json",
            pseudo_label_path=project / "outputs" / "pseudo.csv",
            pseudo_label_report_path=project / "outputs" / "pseudo_report.json",
            pseudo_training_path=project / "outputs" / "train_pseudo.csv",
            grid_report_path=project / "outputs" / "grid.json",
            grid_model_dir=project / "models" / "grid",
            epochs=10.0,
            validation_size=0.2,
            learning_rate=1e-5,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_length=512,
            weight_decay=0.01,
            warmup_ratio=0.06,
            seed=42,
            device="cpu",
            force_preprocess=False,
            force_retrain=False,
            force_predict=False,
            force_pseudo_label=False,
            force_grid_search=False,
            use_class_weights=True,
            gradient_checkpointing=True,
            use_viterbi=True,
            viterbi_lambda=0.75,
            use_pseudo_labels=True,
            pseudo_label_confidence_threshold=0.95,
            pseudo_label_margin_threshold=0.20,
            pseudo_label_min_words=1,
            run_grid_search=False,
            grid_learning_rates=(1e-5,),
            grid_epochs=(3.0,),
            grid_warmup_ratios=(0.06,),
            grid_weight_decays=(0.01,),
            use_duplicate_prior=True,
            duplicate_lambda=1.0,
            optimize_postprocessing=True,
            viterbi_lambda_grid=(0.0, 0.75),
            duplicate_lambda_grid=(0.0, 1.0),
        )

    def test_preprocess_config_preserves_transformer_tokenization_signals(self):
        config = trainer_script.build_preprocess_config()

        self.assertFalse(config.lowercase)
        self.assertFalse(config.strip_accents)
        self.assertTrue(config.keep_digits)
        self.assertFalse(config.remove_stopwords)

    def test_add_transformer_text_column_uses_minimal_bert_cleaning(self):
        df = pd.DataFrame(
            {
                "Id": [1],
                "Body": ["Conclusão do RE 123-DF, art. 102 da Constituição."],
                "Category": [3],
            }
        )

        cleaned = trainer_script.add_transformer_text_column(df)

        self.assertIn(trainer_script.TRANSFORMER_TEXT_COL, cleaned.columns)
        text = cleaned.loc[0, trainer_script.TRANSFORMER_TEXT_COL]
        self.assertIn("Conclusão", text)
        self.assertIn("RE", text)
        self.assertIn("ARTIGO_LEI", text)

    def test_load_preprocessed_frames_uses_existing_cache_when_raw_csvs_are_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            cache_dir = project / "outputs" / "cache"
            cache_dir.mkdir(parents=True)
            (project / "data").mkdir()

            pd.DataFrame(
                {
                    "Id": [1, 2],
                    "Body": ["texto um", "texto dois"],
                    "Body_clean": ["texto um", "texto dois"],
                    "Category": [0, 1],
                }
            ).to_csv(cache_dir / "train_clean.csv", index=False)
            pd.DataFrame({"Id": [3], "Body": ["teste"], "Body_clean": ["teste"]}).to_csv(
                cache_dir / "test_clean.csv",
                index=False,
            )

            bundle = trainer_script.load_preprocessed_frames(
                project_dir=project,
                data_dir=project / "data",
                output_dir=project / "outputs",
                config=trainer_script.build_preprocess_config(),
                force=False,
            )

            self.assertEqual(bundle["cache_status"], "loaded_cache_only")
            self.assertEqual(bundle["train_clean"]["Category"].tolist(), [0, 1])
            self.assertEqual(bundle["test_clean"]["Id"].tolist(), [3])

    def test_select_training_frame_prefers_modeling_cache_and_keeps_valid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            pd.DataFrame(
                {
                    "Id": [10, 11, 12],
                    "Body_clean": ["a", "b", "c"],
                    "Category": [0, 1, -1],
                }
            ).to_csv(cache_dir / "train_modeling.csv", index=False)
            train_clean = pd.DataFrame({"Id": [1], "Body_clean": ["x"], "Category": [2]})

            training, source = trainer_script.select_training_frame(
                cache_dir=cache_dir,
                train_clean=train_clean,
                target_col="Category",
            )

            self.assertEqual(source, str(cache_dir / "train_modeling.csv"))
            self.assertEqual(training["Id"].tolist(), [10, 11])
            self.assertEqual(training["Category"].tolist(), [0, 1])

    def test_align_probabilities_maps_original_labels_to_sorted_class_columns(self):
        probabilities = np.array(
            [
                [0.80, 0.20],
                [0.10, 0.90],
            ]
        )

        aligned = trainer_script.align_probabilities_to_classes(
            probabilities,
            label_values=[4, 2],
            classes=[0, 2, 4],
        )

        self.assertEqual(aligned.shape, (2, 3))
        np.testing.assert_allclose(aligned[:, 0], np.zeros(2))
        np.testing.assert_allclose(aligned[:, 1], np.array([0.20, 0.90]))
        np.testing.assert_allclose(aligned[:, 2], np.array([0.80, 0.10]))
        np.testing.assert_allclose(aligned.sum(axis=1), np.ones(2))

    def test_compute_classification_metrics_reports_accuracy_and_f1(self):
        metrics = trainer_script.compute_classification_metrics(
            y_true=[0, 1, 1, 2],
            y_pred=[0, 1, 2, 2],
        )

        self.assertEqual(metrics["n_eval"], 4)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f1_macro"], (1.0 + (2 / 3) + (2 / 3)) / 3)
        self.assertIn("classification_report", metrics)

    def test_load_or_train_final_transformer_validates_then_trains_full_data_model(self):
        class FakeTokenizer:
            def save_pretrained(self, path):
                self.saved_path = Path(path)

        class FakeTrainer:
            def __init__(self, log_history=None):
                self.processing_class = FakeTokenizer()
                self.state = type(
                    "FakeState",
                    (),
                    {
                        "best_metric": 0.81,
                        "best_model_checkpoint": "checkpoint-20",
                        "log_history": log_history or [],
                    },
                )()

            def save_model(self, path):
                self.saved_model_path = Path(path)

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            config = self._make_config(project)
            experiment_config = trainer_script.build_experiment_config(config.output_dir, config.seed)
            train_df = pd.DataFrame(
                {
                    "Id": [1, 2, 3, 4, 5],
                    "Body_clean": ["a", "b", "c", "d", "e"],
                    "Body_transformer": ["a", "b", "c", "d", "e"],
                    "Category": [0, 1, 2, 3, 4],
                }
            )
            validation_trainer = FakeTrainer(
                [
                    {"epoch": 1.0, "eval_f1_macro": 0.50, "eval_accuracy": 0.60},
                    {"epoch": 2.0, "eval_f1_macro": 0.81, "eval_accuracy": 0.70},
                ]
            )
            final_trainer = FakeTrainer()

            with patch.object(
                trainer_script,
                "fine_tune_transformer_classifier",
                side_effect=[validation_trainer, final_trainer],
            ) as fine_tune:
                trainer, validation_summary = trainer_script.load_or_train_final_transformer(
                    config,
                    experiment_config,
                    train_df,
                )

            self.assertIs(trainer, final_trainer)
            self.assertEqual(fine_tune.call_count, 2)
            validation_kwargs = fine_tune.call_args_list[0].kwargs
            final_kwargs = fine_tune.call_args_list[1].kwargs
            self.assertEqual(validation_kwargs["validation_size"], 0.2)
            self.assertEqual(final_kwargs["validation_size"], 0.0)
            self.assertEqual(final_kwargs["num_train_epochs"], 2.0)
            self.assertEqual(validation_summary["best_metric"], 0.81)
            self.assertEqual(validation_summary["best_epoch"], 2.0)

    def test_select_pseudo_labeled_rows_filters_by_confidence_and_margin(self):
        unlabeled = pd.DataFrame(
            {
                "Id": [10, 11, 12],
                "Body_clean": ["texto juridico forte", "texto juridico fraco", "x"],
                "Category": [-1, -1, -1],
            }
        )
        probabilities = np.array(
            [
                [0.01, 0.96, 0.01, 0.01, 0.01],
                [0.10, 0.77, 0.13, 0.00, 0.00],
                [0.99, 0.01, 0.00, 0.00, 0.00],
            ]
        )

        selected, report = trainer_script.select_pseudo_labeled_rows(
            unlabeled,
            probabilities,
            label_values=[0, 1, 2, 3, 4],
            classes=[0, 1, 2, 3, 4],
            text_col="Body_clean",
            target_col="Category",
            confidence_threshold=0.95,
            margin_threshold=0.20,
            min_words=2,
            source="unit-test",
        )

        self.assertEqual(selected["Id"].tolist(), [10])
        self.assertEqual(selected["Category"].tolist(), [1])
        self.assertTrue(selected["IsPseudoLabel"].all())
        self.assertEqual(report["selected"], 1)
        self.assertEqual(report["rejected_by_length"], 1)

    def test_build_training_frame_with_pseudo_labels_keeps_traceability(self):
        train = pd.DataFrame(
            {
                "Id": [1, 2],
                "Body_clean": ["a", "b"],
                "Category": [0, 1],
            }
        )
        pseudo = pd.DataFrame(
            {
                "Id": [3],
                "Body_clean": ["c"],
                "Category": [2],
                "IsPseudoLabel": [True],
                "PseudoConfidence": [0.98],
            }
        )

        combined = trainer_script.build_training_frame_with_pseudo_labels(
            train,
            pseudo,
            target_col="Category",
        )

        self.assertEqual(combined["Id"].tolist(), [1, 2, 3])
        self.assertEqual(combined["Category"].tolist(), [0, 1, 2])
        self.assertEqual(combined["IsPseudoLabel"].tolist(), [False, False, True])
        self.assertEqual(combined.loc[combined["Id"] == 3, "PseudoConfidence"].item(), 0.98)

    def test_parse_float_grid_accepts_comma_separated_values(self):
        self.assertEqual(trainer_script.parse_float_grid("1e-5, 2e-5"), (1e-5, 2e-5))
        self.assertEqual(trainer_script.parse_float_grid(""), ())

    def test_default_hyperparameters_match_attached_best_grid(self):
        args = trainer_script.parse_args([])

        self.assertEqual(args.epochs, 5.0)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.warmup_ratio, 0.0)
        self.assertEqual(args.weight_decay, 0.01)

    def test_optimize_postprocessing_on_validation_selects_best_text_only_settings(self):
        class FakeTrainer:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            config = replace(self._make_config(project), validation_size=0.5)
            experiment_config = trainer_script.build_experiment_config(config.output_dir, config.seed)
            train_df = pd.DataFrame(
                {
                    "Id": list(range(10)),
                    "Body_transformer": [
                        "acordao turma",
                        "acordao plenario",
                        "agravo are",
                        "agravo recurso",
                        "despacho intime",
                        "despacho vista",
                        "recurso extraordinario",
                        "recurso geral",
                        "sentenca procedente",
                        "sentenca improcedente",
                    ],
                    "Category": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                }
            )
            probabilities = np.array(
                [
                    [0.05, 0.80, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.80, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.80, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.80],
                    [0.80, 0.05, 0.05, 0.05, 0.05],
                ]
            )

            with patch.object(
                trainer_script,
                "predict_transformer_probabilities",
                return_value=(probabilities, [0, 1, 2, 3, 4]),
            ):
                summary = trainer_script.optimize_postprocessing_on_validation(
                    config,
                    experiment_config,
                    FakeTrainer(),
                    train_df,
                    classes=[0, 1, 2, 3, 4],
                )

            self.assertTrue(summary["enabled"])
            self.assertIn(summary["duplicate_lambda"], config.duplicate_lambda_grid)
            self.assertIn(summary["viterbi_lambda"], config.viterbi_lambda_grid)
            self.assertIn("base_argmax_f1_macro", summary)


if __name__ == "__main__":
    unittest.main()
