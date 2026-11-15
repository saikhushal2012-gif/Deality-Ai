from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_deality_match import train


class DealityMatchTrainingTests(unittest.TestCase):
    def test_grouped_training_reports_new_and_original_data_separately(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "deality-match.joblib"
            result = train(model_path=model_path)
            self.assertTrue(model_path.is_file())
        self.assertEqual((result["scenario_count"], result["candidate_count"]), (29, 71))
        self.assertEqual((result["original_pilot_top1_correct"], result["original_pilot_scenarios"]), (0, 3))
        self.assertEqual((result["additional_laptop_scenarios"], result["additional_laptop_top1_correct"]), (11, 6))
        self.assertEqual((result["additional_other_scenarios"], result["additional_other_top1_correct"]), (15, 12))
        self.assertEqual((result["grouped_top1_correct"], result["grouped_top1_total"]), (15, 29))
        self.assertAlmostEqual(result["grouped_candidate_accuracy"], 0.5211267605633803)
        self.assertGreaterEqual(result["grouped_candidate_accuracy"], 0)
        self.assertLessEqual(result["grouped_candidate_accuracy"], 1)
        self.assertEqual(result["grouped_top1_total"], 29)


if __name__ == "__main__":
    unittest.main()
