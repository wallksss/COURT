import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_large_bert as script  # noqa: E402


class TrainLargeBertScriptTests(unittest.TestCase):
    def test_load_and_preprocess_data_rebuilds_from_raw_csvs_and_drops_unlabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            data_dir = project / "data"
            data_dir.mkdir()
            pd.DataFrame(
                {
                    "Id": [1, 2, 3],
                    "Body": ["conclusÃ£o ART. 102", "sentenca   procedente", "sem rotulo"],
                    "Category": [0, 4, -1],
                }
            ).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame({"Id": [4], "Body": ["recurso extraordinÃ¡rio"]}).to_csv(
                data_dir / "test.csv",
                index=False,
            )

            train_df, test_df, report = script.load_and_preprocess_data(
                data_dir=data_dir,
                preprocess_config=script.build_preprocess_config(),
            )

            self.assertEqual(train_df["Id"].tolist(), [1, 2])
            self.assertEqual(train_df["Category"].tolist(), [0, 4])
            self.assertIn("Body_clean", train_df.columns)
            self.assertIn("artigo_lei", train_df.loc[0, "Body_clean"])
            self.assertIn("recurso extraordinario", test_df.loc[0, "Body_clean"])
            self.assertEqual(report["raw_train_rows"], 3)
            self.assertEqual(report["final_train_rows"], 2)
            self.assertEqual(report["dropped_unlabeled_rows"], 1)

    def test_cuda_defaults_are_memory_conservative_for_large_bert(self):
        profile = script.default_hyperparameters_for_device("cuda")

        self.assertEqual(profile["batch_size"], 4)
        self.assertEqual(profile["gradient_accumulation_steps"], 4)
        self.assertEqual(profile["effective_batch_size"], 16)
        self.assertEqual(profile["learning_rate"], 1e-5)
        self.assertEqual(profile["epochs"], 4.0)

    def test_head_tail_tokenization_keeps_document_start_and_end(self):
        class FakeTokenizer:
            cls_token_id = 101
            sep_token_id = 102

            def __call__(self, text, add_special_tokens=False, truncation=False):
                return {"input_ids": [int(token[1:]) for token in text.split()]}

            def num_special_tokens_to_add(self, pair=False):
                return 2

            def prepare_for_model(self, input_ids, truncation=False, max_length=None):
                return {"input_ids": [101, *input_ids, 102], "attention_mask": [1] * (len(input_ids) + 2)}

        encoded = script.tokenize_head_tail_text(
            " ".join(f"t{i}" for i in range(10)),
            FakeTokenizer(),
            max_length=8,
        )

        self.assertEqual(encoded["input_ids"], [101, 0, 1, 2, 3, 8, 9, 102])

    def test_num_labels_from_logits_does_not_require_model_config(self):
        class DataParallelLike:
            pass

        class FakeLogits:
            shape = (8, 5)

        self.assertEqual(script.num_labels_from_logits(DataParallelLike(), FakeLogits()), 5)


if __name__ == "__main__":
    unittest.main()
