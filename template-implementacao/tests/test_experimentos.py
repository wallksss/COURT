import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import experimentos  # noqa: E402


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
