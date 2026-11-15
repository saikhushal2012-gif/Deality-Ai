import unittest

from comparewise.schemas import Product
from comparewise.recommendation import recommend


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.products = [
            Product("Budget tower", "Computers", 700, performance=58, quality=70,
                    rating=4.2, features_score=55),
            Product("Gaming bundle", "Computers", 1100, performance=75, quality=78,
                    rating=4.5, features_score=80,
                    included_items=["monitor", "keyboard", "mouse", "headset"]),
            Product("Power tower", "Computers", 1250, performance=96, quality=88,
                    rating=4.6, features_score=91),
        ]

    def test_bundle_fit_changes_with_needs(self):
        bundle = recommend(
            self.products,
            priorities={"bundle": 1},
            setup_needs=["monitor", "keyboard", "mouse"],
        )
        performance = recommend(self.products, priorities={"performance": 1})
        self.assertEqual(bundle["best_overall"], "Gaming bundle")
        self.assertEqual(performance["best_overall"], "Power tower")

    def test_missing_values_are_accepted(self):
        result = recommend([Product("Unknown", "Laptop")])
        self.assertEqual(result["ranked"][0]["name"], "Unknown")
        self.assertTrue(result["ranked"][0]["missing_fields"])

    def test_mixed_product_types_are_rejected(self):
        products = [
            Product("Everyday watch", "Watches", 100),
            Product("Gaming computer", "Computers", 1000),
        ]
        with self.assertRaisesRegex(ValueError, "same type or purpose"):
            recommend(products)

    def test_product_name_cannot_mislabel_an_obviously_different_type(self):
        products = [
            Product("Everyday watch", "Watches", 100),
            Product("Gaming desktop computer", "Watches", 1000),
        ]
        with self.assertRaisesRegex(ValueError, "appears to be computers"):
            recommend(products)

    def test_common_type_aliases_can_be_compared(self):
        products = [
            Product("Watch A", "watch", 100),
            Product("Watch B", "Watches", 150),
        ]
        result = recommend(products)
        self.assertEqual(len(result["ranked"]), 2)

    def test_direct_inputs_change_the_winner_without_model_training(self):
        products = [
            Product("Affordable option", "Watches", 100, performance=60, quality=65),
            Product("Higher-spec option", "Watches", 180, performance=95, quality=92),
        ]
        budget_pick = recommend(products, priorities={"price": 1})
        quality_pick = recommend(products, priorities={"quality": 1})
        self.assertEqual(budget_pick["best_overall"], "Affordable option")
        self.assertEqual(quality_pick["best_overall"], "Higher-spec option")
        self.assertIn("no model training", budget_pick["model"])

    def test_no_comparable_inputs_does_not_invent_a_winner(self):
        products = [Product("Unknown A", "Watches"), Product("Unknown B", "Watches")]
        result = recommend(products)
        self.assertIsNone(result["best_overall"])


if __name__ == "__main__":
    unittest.main()
