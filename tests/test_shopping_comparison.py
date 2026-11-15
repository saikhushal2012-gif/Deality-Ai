import copy
import math
import unittest

from comparewise.shopping_comparison import (
    PRIORITIES,
    SPEC_FOCUSES,
    comparison_compatibility,
    comparison_rows,
    recommend_many,
    recommend_products,
)


class CompatibilityTests(unittest.TestCase):
    def test_recognized_aliases_and_specific_purposes(self):
        cases = [
            ("wristwatch", "smart watches", "watch"),
            ("laptops", "desktop computer", "computer"),
            ("PC", "computer", "computer"),
            ("hooded sweatshirt", "hoodies", "hoodie"),
            ("cell phones", "smartphone", "smartphone"),
            ("guitars", "guitar", "guitar"),
            ("washing machines", "clothes washer", "washing machine"),
            ("refrigerator", "fridges", "refrigerator"),
        ]
        for first, second, kind in cases:
            with self.subTest(first=first, second=second):
                result = comparison_compatibility({"category": first}, {"category": second})
                self.assertEqual(result["status"], "compatible")
                self.assertEqual(result["types"], [kind, kind])
                self.assertEqual(result["evidence"][0]["source"], "category")

    def test_incompatible_pairs_are_blocked_even_if_confirmed(self):
        for first, second in [("watch", "laptop"), ("hoodie", "smartphone"),
                              ("washing machine", "refrigerator"), ("guitar", "blender"),
                              ("television", "camera")]:
            with self.subTest(first=first, second=second):
                products = [{"category": first}, {"category": second}]
                self.assertEqual(comparison_compatibility(*products)["status"], "incompatible")
                for confirmed in (False, True):
                    with self.assertRaisesRegex(ValueError, "cannot be compared"):
                        comparison_rows(products, confirmed=confirmed)
                    with self.assertRaisesRegex(ValueError, "cannot be compared"):
                        recommend_products(*products, "Lower price", confirmed=confirmed)

    def test_uncertain_pairs_require_external_confirmation(self):
        pairs = [
            ({"category": "guitar"}, {"category": "piano"}),
            ({"category": "appliances"}, {"category": "appliances"}),
            ({"name": "Mystery model 123"}, {"category": "watch"}),
            ({"name": "laptop bag"}, {"category": "laptop"}),
            ({"name": "watch and laptop combo"}, {"category": "watch"}),
            ({"category": "watch", "name": "Laptop computer"}, {"category": "watch"}),
            ({"name": "Notebook"}, {"name": "Notebook"}),
            ({"description": "laptop"}, {"name": "laptop"}),
        ]
        for first, second in pairs:
            with self.subTest(first=first, second=second):
                result = comparison_compatibility(first, second)
                self.assertEqual(result["status"], "uncertain")
                self.assertIn("confirmation", result["reason"])
                with self.assertRaisesRegex(ValueError, "confirmation"):
                    comparison_rows([first, second])
                with self.assertRaisesRegex(ValueError, "confirmation"):
                    recommend_products(first, second, "Lower price")
                self.assertTrue(comparison_rows([first, second], confirmed=True))
                self.assertIsNone(recommend_products(first, second, "Lower price", confirmed=True)["winner_index"])
        self.assertEqual(comparison_compatibility({"category": "appliances"}, {})["types"], [None, None])

    def test_recognized_name_evidence_and_unrecognized_description(self):
        result = comparison_compatibility({"name": "Acme laptop computer"}, {"name": "Brand gaming PC"})
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["evidence"][0]["source"], "name")
        result = comparison_compatibility({"name": "laptop", "description": "Works with phone"}, {"name": "desktop"})
        self.assertEqual(result["status"], "compatible")

    def test_user_confirmation_must_be_boolean_true(self):
        with self.assertRaises(ValueError):
            comparison_rows([{}, {}], confirmed="yes")


class ComparisonRowsTests(unittest.TestCase):
    def test_three_product_rows_keep_dynamic_specification_columns(self):
        products = [
            {"name": "Watch A", "category": "watch", "price": 50, "currency": "USD", "rating": 3,
             "specifications": {"Battery": "2 days"}},
            {"name": "Watch B", "category": "watch", "price": 100, "currency": "USD", "rating": 5,
             "specifications": {"Battery": "5 days"}},
            {"name": "Watch C", "category": "watch", "price": 75, "currency": "USD", "rating": 4,
             "specifications": {"Water resistance": "50 m"}},
        ]
        rows = {row["Detail"]: row for row in comparison_rows(products)}
        self.assertEqual(rows["Battery"], {"Detail": "Battery", "Option 1": "2 days", "Option 2": "5 days", "Option 3": "Not listed"})
        self.assertEqual(rows["Water resistance"]["Option 3"], "50 m")
        self.assertEqual(rows["Price"]["Option 2"], "USD 100")

    def test_generic_fields_dynamic_specs_seller_and_missing(self):
        products = [
            {"category": "laptop", "name": "First", "url": "https://shop.example/1", "image_url": "https://img.example/1",
             "price": 50, "currency": "usd", "rating": 4.5, "review_count": 0, "seller": "Shop A",
             "description": "A description", "features": ["Fast", "Quiet"],
             "specifications": {" RAM_Size ": "16 GB", "RAM size": "32 GB", "Feature 1": "Fast", "Feature 2": "Light", "Battery": "10h"}},
            {"category": "desktop", "name": "Second", "price_display": "€100", "currency": "EUR", "rating": 4,
             "seller": "Shop B", "features": ["Powerful"], "specifications": {"ram_size": "64 GB", "Weight": "5 kg", "feature_1": "Compact"}},
        ]
        before = copy.deepcopy(products)
        rows = comparison_rows(products)
        by_detail = {row["Detail"]: row for row in rows}
        self.assertEqual(products, before)
        self.assertEqual(by_detail["Name"]["Option 2"], "Second")
        self.assertEqual(by_detail["Seller"]["Option 1"], "Shop A")
        self.assertEqual(by_detail["Seller"]["Option 2"], "Shop B")
        self.assertEqual(by_detail["Price"]["Option 1"], "USD 50")
        self.assertEqual(by_detail["Price"]["Option 2"], "€100")
        self.assertEqual(by_detail["Reviews"]["Option 1"], "0")
        self.assertEqual(by_detail["Reviews"]["Option 2"], "Not listed")
        self.assertEqual(by_detail[" RAM_Size "]["Option 1"], "16 GB\n32 GB")
        self.assertEqual(by_detail[" RAM_Size "]["Option 2"], "64 GB")
        self.assertEqual(by_detail["Weight"]["Option 1"], "Not listed")
        self.assertEqual(by_detail["Battery"]["Option 2"], "Not listed")
        self.assertEqual(by_detail["Features"]["Option 1"], "Fast\nQuiet\nLight")
        self.assertEqual(by_detail["Features"]["Option 2"], "Powerful\nCompact")
        self.assertNotIn("Feature 1", by_detail)
        self.assertNotIn("ASIN", by_detail)
        self.assertNotIn("Brand", by_detail)
        self.assertEqual(by_detail["URL"]["Option 1"], "https://shop.example/1")
        self.assertEqual(by_detail["Image URL"]["Option 2"], "Not listed")

    def test_missing_and_invalid_values(self):
        products = [{"category": "watch", "price": 5, "rating": 8, "review_count": 1.5,
                     "features": "One feature", "specifications": {"Material": None}},
                    {"category": "watch", "price": 10, "currency": "USD", "features": None, "specifications": None}]
        rows = {r["Detail"]: r for r in comparison_rows(products)}
        for label in ("Price", "Rating", "Reviews", "Seller", "Material"):
            self.assertEqual(rows[label]["Option 1"], "Not listed")
        self.assertEqual(rows["Features"]["Option 1"], "One feature")
        self.assertEqual(rows["Features"]["Option 2"], "Not listed")

    def test_requires_two_to_five_products(self):
        for products in ([], [{}], [{}] * 6):
            with self.subTest(products=products), self.assertRaisesRegex(ValueError, "two and five"):
                comparison_rows(products)


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.first = {"name": "Watch A", "price": 50, "currency": "USD", "rating": 3, "review_count": 20}
        self.second = {"name": "Watch B", "price": 100, "currency": "usd", "rating": 5, "review_count": 100}

    def test_all_numeric_priorities(self):
        cases = {"Balanced value": (0, [80, 75]), "Lower price": (0, [100, 50]),
                 "Higher customer rating": (1, [60, 100]), "More customer reviews": (1, [20, 100]),
                 "Price-focused value": (0, [88, 65]), "Rating-focused value": (1, [72, 85])}
        self.assertEqual(set(cases), set(PRIORITIES))
        for priority, (winner, scores) in cases.items():
            with self.subTest(priority=priority):
                result = recommend_products(self.first, self.second, priority)
                self.assertEqual(result["winner_index"], winner)
                for actual, expected in zip(result["scores"], scores):
                    self.assertAlmostEqual(actual, expected)
                self.assertIn("effective weight", result["explanation"])
                self.assertIn("Option 1:", result["score_text"])
        self.assertIn("not a measure of quality", recommend_products(self.first, self.second, "More customer reviews")["explanation"])

    def test_ties_and_zero_values(self):
        zero = {"category": "hoodie", "price": 0, "currency": "USD", "rating": 0, "review_count": 0}
        for priority in PRIORITIES:
            with self.subTest(priority=priority):
                result = recommend_products(zero, dict(zero), priority)
                self.assertIsNone(result["winner_index"])
                self.assertEqual(result["scores"][0], result["scores"][1])
                self.assertIn("tied", result["explanation"])
        result = recommend_products(zero, {**zero, "price": 5}, "Lower price")
        self.assertEqual(result["scores"], [100, 0])

    def test_currencies_and_missing_data_exclude_and_renormalize(self):
        for currency in ("GBP", "", None, "$", "US Dollars"):
            with self.subTest(currency=currency):
                second = {**self.second, "currency": currency}
                price = recommend_products(self.first, second, "Lower price")
                self.assertEqual(price["scores"], [None, None])
                self.assertIn("currenc", price["explanation"])
                balanced = recommend_products(self.first, second, "Balanced value")
                self.assertEqual(balanced["scores"], [60, 100])
                self.assertIn("effective weight 100%", balanced["explanation"])
        result = recommend_products({**self.first, "rating": None}, self.second, "Balanced value")
        self.assertEqual(result["scores"], [100, 50])
        self.assertIn("Excluded for both options: rating", result["explanation"])

    def test_invalid_values_do_not_create_winners(self):
        cases = [("price", -1, "Lower price"), ("price", math.nan, "Lower price"),
                 ("rating", 5.1, "Higher customer rating"), ("rating", math.inf, "Higher customer rating"),
                 ("review_count", -1, "More customer reviews"), ("review_count", 1.5, "More customer reviews"),
                 ("review_count", True, "More customer reviews")]
        for field, value, priority in cases:
            with self.subTest(field=field, value=value):
                result = recommend_products({**self.first, field: value}, self.second, priority)
                self.assertEqual(result["scores"], [None, None])
                self.assertIsNone(result["winner_index"])
                self.assertIn("insufficient comparable data", result["explanation"])

    def test_spec_focus_is_evidence_only(self):
        first = {**self.first, "specifications": {"Battery": "10 hours"}}
        second = {**self.second, "specifications": {"Battery": "20 hours"}}
        result = recommend_products(first, second, "Battery & power")
        self.assertEqual(result["scores"], [None, None])
        self.assertIsNone(result["winner_index"])
        self.assertIn("10 hours", result["explanation"])
        self.assertIn("20 hours", result["explanation"])
        for priority in SPEC_FOCUSES:
            with self.subTest(priority=priority):
                self.assertIsNone(recommend_products(first, second, priority)["winner_index"])

    def test_recommend_many_ranks_three_options_and_blocks_currency_mixing(self):
        products = [
            {"name": "Watch A", "category": "watch", "price": 50, "currency": "USD", "rating": 3},
            {"name": "Watch B", "category": "watch", "price": 100, "currency": "USD", "rating": 5},
            {"name": "Watch C", "category": "watch", "price": 75, "currency": "USD", "rating": 4},
        ]
        cheap = recommend_many(products, "Lower price")
        self.assertEqual(cheap["winner_index"], 0)
        self.assertEqual(len(cheap["scores"]), 3)
        self.assertIn("Option 3:", cheap["score_text"])
        rated = recommend_many(products, "Higher customer rating")
        self.assertEqual(rated["winner_index"], 1)
        products[2]["currency"] = "EUR"
        mixed = recommend_many(products, "Balanced value")
        self.assertEqual(mixed["winner_index"], 1)
        self.assertIn("different currencies", mixed["explanation"])

    def test_unknown_priority(self):
        with self.assertRaisesRegex(ValueError, "Unknown recommendation priority"):
            recommend_products(self.first, self.second, "Mystery")


if __name__ == "__main__":
    unittest.main()