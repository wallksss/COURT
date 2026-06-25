import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import analise_exploratoria, experimentos, notebook_support, preprocessamento  # noqa: E402


class CleaningTests(unittest.TestCase):
    def test_clean_body_fixes_mojibake_wrapper_and_preserves_legal_tokens(self):
        cleaned = preprocessamento.clean_body('{"conclusÃ£o   ARTIGO_102\nEMAIL"}')

        self.assertEqual(cleaned, "conclus\u00e3o ARTIGO_102 EMAIL")

    def test_cached_preprocessed_data_reuses_existing_csvs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "outputs").mkdir()
            pd.DataFrame(
                {
                    "Id": [1, 2],
                    "Body": ["conclusÃ£o", "sentenca"],
                    "Category": [0, -1],
                }
            ).to_csv(project / "data" / "train.csv", index=False)
            pd.DataFrame({"Id": [3], "Body": ["teste"]}).to_csv(project / "data" / "test.csv", index=False)

            first = notebook_support.cached_preprocessed_data(project_dir=project)
            second = notebook_support.cached_preprocessed_data(project_dir=project)

            self.assertEqual(first["cache_status"], "rebuilt")
            self.assertEqual(second["cache_status"], "loaded")
            self.assertTrue((project / "outputs" / "cache" / "train_clean.csv").exists())


class DuplicateAndEnsembleTests(unittest.TestCase):
    def test_duplicate_text_report_counts_conflicts_and_test_overlap(self):
        train = pd.DataFrame(
            {
                "Body_clean": ["a", "a", "b", "b", "c"],
                "Category": [0, 0, 1, 2, 2],
            }
        )
        test = pd.DataFrame({"Body_clean": ["a", "x"]})

        report = analise_exploratoria.duplicate_text_report(train, test, text_col="Body_clean")

        self.assertEqual(report["summary"]["n_unique_texts"], 3)
        self.assertEqual(report["summary"]["n_duplicate_rows"], 2)
        self.assertEqual(report["summary"]["n_conflicting_texts"], 1)
        self.assertEqual(report["summary"]["n_same_label_duplicate_texts"], 1)
        self.assertEqual(report["summary"]["n_test_rows_seen_in_train"], 1)

    def test_duplicate_prior_uses_reference_counts_with_smoothing(self):
        reference = pd.DataFrame(
            {
                "Body_clean": ["texto a", "texto a", "texto b"],
                "Category": [0, 0, 1],
            }
        )

        probs = experimentos.duplicate_prior_probabilities(
            reference,
            target_texts=["texto a", "texto novo"],
            classes=[0, 1, 2],
            alpha=1.0,
        )

        np.testing.assert_allclose(probs[0], np.array([3 / 5, 1 / 5, 1 / 5]))
        np.testing.assert_allclose(probs[1], np.array([1 / 3, 1 / 3, 1 / 3]))

    def test_duplicate_prior_combines_with_text_probs_in_log_space(self):
        text_probs = np.array([[0.40, 0.60]])
        duplicate_probs = np.array([[0.90, 0.10]])

        combined = experimentos.combine_with_duplicate_prior(
            text_probs,
            duplicate_probs,
            lambda_dup=2.0,
        )

        self.assertEqual(combined.argmax(axis=1).tolist(), [0])
        np.testing.assert_allclose(combined.sum(axis=1), np.array([1.0]))

    def test_optimize_ensemble_weights_selects_informative_model(self):
        y_true = np.array([0, 1, 1, 0])
        good = np.array(
            [
                [0.95, 0.05],
                [0.05, 0.95],
                [0.10, 0.90],
                [0.90, 0.10],
            ]
        )
        bad = good[:, ::-1]

        result = experimentos.optimize_ensemble_weights(
            {"good": good, "bad": bad},
            y_true,
            classes=[0, 1],
            step=0.5,
        )

        self.assertEqual(result["weights"]["good"], 1.0)
        self.assertEqual(result["weights"]["bad"], 0.0)
        self.assertEqual(result["f1_macro"], 1.0)


class OutOfFoldTests(unittest.TestCase):
    def test_make_oof_probabilities_aligns_columns_and_test_shape(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        train = pd.DataFrame(
            {
                "Body_clean": [
                    "acordao turma alfa",
                    "acordao pleno alfa",
                    "recurso extraordinario beta",
                    "recurso beta geral",
                    "sentenca juiz gama",
                    "sentenca procedente gama",
                ],
                "Category": [0, 0, 1, 1, 2, 2],
            }
        )
        test = pd.DataFrame({"Body_clean": ["alfa", "beta", "gama"]})
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=500, random_state=0)),
            ]
        )

        oof, test_probs = experimentos.make_oof_probabilities(
            train,
            test,
            model,
            classes=[0, 1, 2],
            cv_folds=2,
            random_state=0,
        )

        self.assertEqual(oof.shape, (6, 3))
        self.assertEqual(test_probs.shape, (3, 3))
        np.testing.assert_allclose(oof.sum(axis=1), np.ones(6))
        np.testing.assert_allclose(test_probs.sum(axis=1), np.ones(3))

    def test_make_oof_probabilities_refits_full_train_for_test_by_default(self):
        from sklearn.base import BaseEstimator, ClassifierMixin

        class FitSizeClassifier(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                self.classes_ = np.array(sorted(set(y)))
                self.fit_size_ = len(X)
                return self

            def predict_proba(self, X):
                p0 = self.fit_size_ / 10.0
                return np.tile(np.array([[p0, 1.0 - p0]]), (len(X), 1))

        train = pd.DataFrame(
            {
                "Body_clean": ["a0", "a1", "b0", "b1", "a2", "b2"],
                "Category": [0, 0, 1, 1, 0, 1],
            }
        )
        test = pd.DataFrame({"Body_clean": ["x"]})

        _, test_probs = experimentos.make_oof_probabilities(
            train,
            test,
            FitSizeClassifier(),
            classes=[0, 1],
            cv_folds=2,
            random_state=0,
        )

        np.testing.assert_allclose(test_probs, np.array([[0.6, 0.4]]))


class TransformerTokenizationTests(unittest.TestCase):
    def test_head_tail_tokenization_keeps_beginning_and_ending_tokens(self):
        class FakeTokenizer:
            cls_token_id = 101
            sep_token_id = 102

            def __call__(self, text, add_special_tokens=False, truncation=False):
                ids = [int(token[1:]) for token in text.split()]
                if add_special_tokens:
                    ids = [self.cls_token_id, *ids, self.sep_token_id]
                return {"input_ids": ids}

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def prepare_for_model(self, input_ids, truncation=False, max_length=None):
                ids = [self.cls_token_id, *input_ids, self.sep_token_id]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        encoded = experimentos.tokenize_head_tail_text(
            " ".join(f"t{i}" for i in range(10)),
            FakeTokenizer(),
            max_length=8,
            head_tokens=4,
            tail_tokens=2,
        )

        self.assertEqual(encoded["input_ids"], [101, 0, 1, 2, 3, 8, 9, 102])

    def test_head_tail_tokenization_supports_tokenizer_without_prepare_for_model(self):
        class LegacyBertTokenizer:
            def __call__(self, text, add_special_tokens=False, truncation=False):
                return {"input_ids": [int(token[1:]) for token in text.split()]}

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
                return [101, *token_ids_0, 102]

            def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
                return [0] * (len(token_ids_0) + 2)

        encoded = experimentos.tokenize_head_tail_text(
            " ".join(f"t{i}" for i in range(10)),
            LegacyBertTokenizer(),
            max_length=8,
            head_tokens=4,
            tail_tokens=2,
        )

        self.assertEqual(encoded["input_ids"], [101, 0, 1, 2, 3, 8, 9, 102])
        self.assertEqual(encoded["attention_mask"], [1] * 8)
        self.assertEqual(encoded["token_type_ids"], [0] * 8)

    def test_head_tail_default_scales_with_window_size(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False, truncation=False):
                return {"input_ids": list(range(20))}

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def prepare_for_model(self, input_ids, truncation=False, max_length=None):
                return {"input_ids": [101, *input_ids, 102], "attention_mask": [1] * (len(input_ids) + 2)}

        encoded = experimentos.tokenize_head_tail_text("texto longo", FakeTokenizer(), max_length=12)

        self.assertEqual(encoded["input_ids"], [101, 0, 1, 2, 3, 4, 5, 6, 7, 18, 19, 102])

    def test_predict_transformer_probabilities_returns_probs_and_original_labels(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False, truncation=False):
                return {"input_ids": [1, 2, 3]}

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def prepare_for_model(self, input_ids, truncation=False, max_length=None):
                return {"input_ids": [101, *input_ids, 102], "attention_mask": [1] * (len(input_ids) + 2)}

        class FakeConfig:
            id2label = {0: "2", 1: "4"}

        class FakeModel:
            config = FakeConfig()

        class FakePrediction:
            predictions = np.array([[0.0, 2.0], [2.0, 0.0]])

        class FakeTrainer:
            model = FakeModel()
            processing_class = FakeTokenizer()

            def predict(self, dataset):
                return FakePrediction()

        probs, labels = experimentos.predict_transformer_probabilities(
            FakeTrainer(),
            pd.DataFrame({"Body_clean": ["a", "b"]}),
            text_col="Body_clean",
            max_length=8,
        )

        self.assertEqual(labels, [2, 4])
        self.assertEqual(probs.shape, (2, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))


class SequentialViterbiTests(unittest.TestCase):
    def test_viterbi_keeps_known_labels_fixed(self):
        emission_probs = np.array(
            [
                [0.90, 0.10],
                [0.99, 0.01],
                [0.55, 0.45],
            ]
        )
        transition_probs = np.array(
            [
                [0.80, 0.20],
                [0.20, 0.80],
            ]
        )

        decoded = experimentos.viterbi_decode(
            emission_probs,
            transition_probs,
            allowed_label_indices=[None, {1}, None],
        )

        self.assertEqual(decoded[1], 1)

    def test_transductive_viterbi_submission_preserves_original_test_order(self):
        config = experimentos.ExperimentConfig(text_col="Body_clean", target_col="Category", id_col="Id")
        train = pd.DataFrame(
            {
                "Id": [1, 2],
                "Body_clean": ["treino um", "treino dois"],
                "Category": [0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "Id": [30, 10, 20],
                "Body_clean": ["teste trinta", "teste dez", "teste vinte"],
            }
        )
        probs = np.array(
            [
                [0.10, 0.90],
                [0.95, 0.05],
                [0.20, 0.80],
            ]
        )

        submission = experimentos.transductive_viterbi_submission(
            train,
            test,
            probs,
            classes=[0, 1],
            config=config,
            lambda_transition=0.0,
        )

        self.assertEqual(submission["Id"].tolist(), [30, 10, 20])
        self.assertEqual(submission["Category"].tolist(), [1, 0, 1])


class TrainerCompatibilityTests(unittest.TestCase):
    def test_legacy_trainer_does_not_receive_tokenizer_keyword(self):
        tokenizer = object()

        class LegacyTrainer:
            def __init__(self, **kwargs):
                if "tokenizer" in kwargs:
                    raise TypeError("unexpected keyword argument 'tokenizer'")
                self.kwargs = kwargs

        trainer = experimentos._instantiate_trainer_compat(
            LegacyTrainer,
            trainer_kwargs={"model": object(), "args": object(), "tokenizer": tokenizer},
            tokenizer=tokenizer,
        )

        self.assertIs(trainer.processing_class, tokenizer)
        self.assertNotIn("tokenizer", trainer.kwargs)

    def test_num_labels_from_logits_does_not_require_model_config(self):
        class DataParallelLike:
            pass

        logits = np.zeros((4, 5))

        self.assertEqual(experimentos._num_labels_from_logits(DataParallelLike(), logits), 5)

    def test_new_trainer_receives_processing_class_when_supported(self):
        tokenizer = object()

        class NewTrainer:
            def __init__(self, processing_class=None, **kwargs):
                self.processing_class = processing_class
                self.kwargs = kwargs

        trainer = experimentos._instantiate_trainer_compat(
            NewTrainer,
            trainer_kwargs={"model": object(), "args": object(), "tokenizer": tokenizer},
            tokenizer=tokenizer,
        )

        self.assertIs(trainer.processing_class, tokenizer)
        self.assertNotIn("tokenizer", trainer.kwargs)


class PseudoLabelingTests(unittest.TestCase):
    def test_pseudo_labeling_keeps_only_high_confidence_rows(self):
        class ConfidenceModel:
            classes_ = np.array([0, 1, 2])

            def predict_proba(self, texts):
                self.texts = list(texts)
                return np.array(
                    [
                        [0.91, 0.05, 0.04],
                        [0.10, 0.55, 0.35],
                        [0.02, 0.03, 0.95],
                    ]
                )

        unlabeled = pd.DataFrame(
            {
                "Id": [10, 11, 12],
                "Body_clean": ["acordao turma", "texto duvidoso", "sentenca juiz"],
                "Category": [-1, -1, -1],
            }
        )

        selected, report = experimentos.pseudo_label_unlabeled_samples(
            unlabeled,
            ConfidenceModel(),
            min_confidence=0.90,
        )

        self.assertEqual(selected["Id"].tolist(), [10, 12])
        self.assertEqual(selected["Category"].tolist(), [0, 2])
        self.assertEqual(report["selected"], 2)
        self.assertAlmostEqual(report["coverage"], 2 / 3)

    def test_label_audit_flags_confident_disagreements(self):
        class AuditModel:
            classes_ = np.array([0, 1, 2])

            def predict_proba(self, texts):
                return np.array(
                    [
                        [0.93, 0.04, 0.03],
                        [0.03, 0.02, 0.95],
                        [0.20, 0.60, 0.20],
                    ]
                )

        labeled = pd.DataFrame(
            {
                "Id": [20, 21, 22],
                "Body_clean": ["acordao", "sentenca", "recurso"],
                "Category": [0, 1, 1],
            }
        )

        issues = experimentos.detect_label_issues(labeled, AuditModel(), min_confidence=0.90)

        self.assertEqual(issues["Id"].tolist(), [21])
        self.assertEqual(issues["Category"].tolist(), [1])
        self.assertEqual(issues["SuggestedCategory"].tolist(), [2])

    def test_training_set_can_add_pseudo_labels_and_conservative_corrections(self):
        labeled = pd.DataFrame(
            {
                "Id": [30, 31],
                "Body_clean": ["texto um", "texto dois"],
                "Category": [0, 1],
            }
        )
        pseudo = pd.DataFrame(
            {
                "Id": [32],
                "Body_clean": ["texto tres"],
                "Category": [2],
                "PseudoConfidence": [0.96],
            }
        )
        issues = pd.DataFrame(
            {
                "Id": [31],
                "SuggestedCategory": [2],
                "SuggestedConfidence": [0.98],
            }
        )

        training = experimentos.build_training_set_with_pseudo_labels(
            labeled,
            pseudo,
            label_issues=issues,
            correction_min_confidence=0.97,
        )

        self.assertEqual(training.loc[training["Id"] == 31, "Category"].item(), 2)
        self.assertEqual(training.loc[training["Id"] == 31, "OriginalCategory"].item(), 1)
        self.assertTrue(training.loc[training["Id"] == 32, "IsPseudoLabel"].item())


class ChunkingTests(unittest.TestCase):
    def test_word_chunker_preserves_overlap(self):
        chunks = experimentos._split_text_into_word_chunks(
            " ".join(f"w{i}" for i in range(10)),
            max_words=4,
            overlap_words=1,
        )

        self.assertEqual(chunks[0], "w0 w1 w2 w3")
        self.assertEqual(chunks[1], "w3 w4 w5 w6")
        self.assertEqual(chunks[-1], "w6 w7 w8 w9")


if __name__ == "__main__":
    unittest.main()
