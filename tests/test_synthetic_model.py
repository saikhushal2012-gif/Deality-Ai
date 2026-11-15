from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_pilot_model import DEFAULT_DATA, load_labeled_scenarios
from scripts.train_synthetic_model import (
    DEFAULT_SYNTHETIC_DATA,
    VALIDATION_SCENARIO_IDS,
    convert_examples_to_rows,
    group_scenarios,
    train_synthetic_model,
)


class SyntheticModelTests(unittest.TestCase):
    def test_examples_convert_to_existing_labeled_row_schema(self):
        rows = convert_examples_to_rows(DEFAULT_SYNTHETIC_DATA)
        scenarios = group_scenarios(rows)
        self.assertEqual(len(scenarios), 20)
        self.assertEqual(len(rows), 60)
        self.assertEqual({row["product_category"] for row in rows},
                         {"laptop", "watch", "hoodie", "musical_instrument"})
        required_schema = {
            "scenario_id", "shopper_profile", "budget_max", "budget_currency",
            "hard_requirements", "priorities", "product_name", "price",
            "price_currency", "user_preferred", "preference_reason", "review_status",
            "product_category", "product_attributes",
        }
        self.assertTrue(required_schema.issubset(rows[0]))
        self.assertTrue(all(sum(row["user_preferred"] == "yes" for row in candidates) == 1
                            for candidates in scenarios.values()))
        self.assertIn("battery_hours", next(row["product_attributes"] for row in rows if row["scenario_id"] == "synthetic_laptop_001"))

    def test_validation_scenarios_are_held_out_and_metrics_are_separate(self):
        pilot_before = load_labeled_scenarios(DEFAULT_DATA)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "synthetic.joblib"
            result = train_synthetic_model(DEFAULT_SYNTHETIC_DATA, artifact)
            self.assertTrue(artifact.is_file())
        self.assertEqual(result["training_scenarios"], 16)
        self.assertEqual(result["validation_scenarios"], 4)
        self.assertEqual(result["training_candidates"], 48)
        self.assertEqual(result["validation_candidates"], 12)
        self.assertEqual(set(result["validation_scenario_ids"]), VALIDATION_SCENARIO_IDS)
        self.assertIsInstance(result["training_accuracy"], float)
        self.assertIsInstance(result["validation_accuracy"], float)
        self.assertEqual(pilot_before, load_labeled_scenarios(DEFAULT_DATA))


if __name__ == "__main__":
    unittest.main()
