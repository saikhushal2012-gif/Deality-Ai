from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_pilot_model import (
    DEFAULT_ADDITIONAL_DATA, DEFAULT_DATA, DEFAULT_OTHER_DATA, load_additional_scenarios,
    load_deality_match_scenarios, load_labeled_scenarios, recommend_candidates, row_text,
    supports_deality_match_products, supports_laptop_products, train,
)


class PilotModelTests(unittest.TestCase):
    def test_only_explicit_single_winner_scenarios_are_eligible(self):
        scenarios = load_labeled_scenarios(DEFAULT_DATA)
        self.assertEqual(set(scenarios), {"student_laptop_001", "student_laptop_002", "gaming_laptop_006"})
        self.assertTrue(all(sum(row["user_preferred"] == "yes" for row in rows) == 1 for rows in scenarios.values()))
        self.assertTrue(all(row["user_preferred"] in {"yes", "no"} for rows in scenarios.values() for row in rows))

    def test_features_do_not_include_target_or_label_rationale(self):
        row = {"shopper_profile": "student", "product_name": "laptop",
               "user_preferred": "yes", "preference_reason": "private label rationale"}
        text = row_text(row)
        self.assertIn("student", text)
        self.assertNotIn("private label rationale", text)

    def test_pilot_recommender_ranks_laptops_and_rejects_other_categories(self):
        products = [
            {"name": "Laptop Alpha", "category": "Laptop", "price": 600, "currency": "USD",
             "specifications": {"RAM": "16GB", "CPU": "Ryzen 5", "GPU": "Integrated graphics"}},
            {"name": "Laptop Beta", "category": "Laptop", "price": 700, "currency": "USD",
             "specifications": {"RAM": "16GB", "CPU": "Core i5", "GPU": "Integrated graphics"}},
        ]
        self.assertTrue(supports_laptop_products(products))
        result = recommend_candidates(products, "Performance & components", [])
        self.assertIn(result["winner_index"], (0, 1, None))
        self.assertEqual(result["training_scenarios"], 29)
        self.assertEqual(result["candidate_count"], 71)
        self.assertEqual(result["original_pilot_scenarios"], 3)
        self.assertEqual(result["original_pilot_matches"], 0)
        self.assertEqual(result["combined_cv_scenarios"], 29)
        self.assertFalse(supports_laptop_products([products[0], {"name": "Smart watch"}]))

    def test_supported_non_laptop_categories_are_accepted_but_mixed_categories_are_not(self):
        phones = [
            {"name": "Google Pixel 8a", "category": "Smartphone"},
            {"name": "Samsung Galaxy A55 5G", "category": "Smartphone"},
        ]
        self.assertTrue(supports_deality_match_products(phones))
        result = recommend_candidates(phones, "Higher customer rating", [])
        self.assertEqual(result["category"], "smartphone")
        self.assertEqual(result["category_scenarios"], 2)
        self.assertFalse(supports_deality_match_products([phones[0], {"name": "Laptop Alpha"}]))

    def test_six_user_laptop_scenarios_and_fifteen_other_categories_preserve_pilot(self):
        additional = load_additional_scenarios(DEFAULT_ADDITIONAL_DATA)
        other = load_additional_scenarios(DEFAULT_OTHER_DATA)
        combined = load_deality_match_scenarios()
        original = load_labeled_scenarios(DEFAULT_DATA)
        self.assertEqual(len(additional), 11)
        self.assertEqual(sum(len(rows) for rows in additional.values()), 27)
        self.assertEqual(len(other), 15)
        self.assertEqual(sum(len(rows) for rows in other.values()), 35)
        self.assertEqual(len(combined), 29)
        self.assertEqual(set(combined) - set(additional) - set(other), set(original))
        for rows in (*additional.values(), *other.values()):
            self.assertEqual(sum(row["user_preferred"] == "yes" for row in rows), 1)
            self.assertEqual(sum(row["user_preferred"] == "no" for row in rows), len(rows) - 1)
            self.assertTrue(all("not independently verified" in row["source_note"] for row in rows))
        esports = additional["esports_laptop_007"][0]
        self.assertIn("Above stated budget", esports["product_attributes"])
        phone = other["scenario_012_everyday_value_smartphone"][0]
        self.assertEqual(phone["product_category"], "smartphone")

    def test_grouped_evaluation_trains_and_saves_local_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "pilot.joblib"
            result = train(DEFAULT_DATA, model_path=artifact)
            self.assertEqual(result["scenarios"], 3)
            self.assertEqual(result["candidates"], 9)
            self.assertTrue(artifact.is_file())


if __name__ == "__main__":
    unittest.main()