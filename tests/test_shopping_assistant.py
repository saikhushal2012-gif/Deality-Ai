import sys
import unittest
from copy import deepcopy
from types import ModuleType
from unittest.mock import Mock, patch

from comparewise import shopping_comparison
from comparewise.shopping_assistant import answer_shopping_question


class ShoppingAssistantTests(unittest.TestCase):
    def setUp(self):
        self.products = [
            {"name": "Trail Pack A", "url": "https://store.example/a", "image_url": "https://store.example/a.jpg",
             "price": 50, "price_display": "$50", "currency": "USD", "rating": 4.1,
             "review_count": 20, "seller": "Outdoor Shop", "features": ["Rain cover included"],
             "specifications": {"Capacity": "20 L"}, "description": "Day pack"},
            {"name": "Trail Pack B", "url": "https://other.example/b", "image_url": "https://other.example/b.jpg",
             "price": 90, "price_display": "$90", "currency": "USD", "rating": 4.8,
             "review_count": 200, "seller": "Hiking Market", "features": ["Padded straps"],
             "specifications": {"Capacity": "30 L"}, "description": "Hiking pack"},
        ]
        module = ModuleType("comparewise.shopping_comparison")
        module.comparison_compatibility = Mock(return_value={"status": "compatible", "reason": "Compatible."})
        module.SPEC_FOCUSES = {"Size & weight": ("capacity", "weight")}
        module.comparison_rows = Mock(return_value=[{"Detail": "Capacity", "Option 1": "20 L", "Option 2": "30 L"}])
        module.recommend_products = Mock(return_value={"winner_index": 1, "explanation": "Option 2 has the higher comparable score.", "score_text": "Option 2: 96/100"})
        self.module = module
        self.module_patch = patch.dict(sys.modules, {"comparewise.shopping_comparison": module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_listing_evidence_and_sellers_without_invented_capabilities(self):
        original = deepcopy(self.products)
        answer = answer_shopping_question("Which has a rain cover?", self.products)
        self.assertIn("Rain cover included", answer)
        self.assertIn("no matching detail was returned", answer)
        self.assertIn("can't verify details", answer)
        answer = answer_shopping_question("Show ratings", self.products)
        self.assertIn("Outdoor Shop", answer)
        self.assertIn("Hiking Market", answer)
        self.assertNotIn("Amazon", answer)
        self.assertEqual(self.products, original)

    def test_budget_and_feature_are_separate(self):
        answer = answer_shopping_question("Under $60 with rain cover", self.products)
        self.assertIn("Trail Pack A is the only listing", answer)
        self.assertIn("Rain cover included", answer)
        self.assertIn("does not establish a feature match", answer)

    def test_currency_budget_comparisons_and_boundaries(self):
        for currency, marker in (("USD", "$"), ("EUR", "€"), ("GBP", "GBP "), ("CAD", "CAD "), ("INR", "₹")):
            with self.subTest(currency=currency):
                products = deepcopy(self.products)
                for p in products:
                    p["currency"] = currency
                    p["price_display"] = f"{currency} {p['price']}"
                answer = answer_shopping_question(f"Under {marker}90", products)
                self.assertIn("Trail Pack A is the only listing", answer)
                self.assertIn(currency, answer)
                answer = answer_shopping_question(f"Budget {marker}90", products)
                self.assertIn("Both prices", answer)
        self.assertIn("ambiguous", answer_shopping_question("Under 60 JPY", self.products))
        self.assertIn("different currency", answer_shopping_question("Under €60", self.products))

    def test_mixed_unknown_missing_and_invalid_prices_do_not_win(self):
        for change in ({"currency": "GBP"}, {"price": None}, {"price": -1}, {"price": float("nan")}):
            with self.subTest(change=change):
                products = deepcopy(self.products)
                products[1].update(change)
                for question in ("Which is cheaper?", "Under $60 which is best?"):
                    answer = answer_shopping_question(question, products)
                    self.assertNotIn("is the only listing", answer)
                    self.assertNotIn("is the pick", answer)
                    expected = "can't compare" if question.startswith("Which") else "cannot be compared"
                    self.assertIn(expected, answer)

    def test_cheaper_question_answers_from_visible_amounts_without_claiming_currency(self):
        products = deepcopy(self.products)
        products[0].update(price_display="$899.00", price=899, currency=None, name="Acer Nitro")
        products[1].update(price_display="$1,399.00", price=1399, currency=None, name="Laptop B")
        answer = answer_shopping_question("Which is cheaper?", products)
        self.assertIn("Acer Nitro has the lower displayed price", answer)
        self.assertIn("$899.00 vs $1,399.00", answer)
        self.assertNotIn("currency", answer)
        self.assertNotIn("cannot be compared", answer)
        self.assertNotIn("balanced value", answer)
        self.assertLessEqual(len(answer.split()), 12)

    def test_more_expensive_question_names_product_and_prices(self):
        products = deepcopy(self.products)
        products[0].update(price_display="$899.00", price=899, currency=None, name="Acer Nitro")
        products[1].update(price_display="$2,149.99", price=2149.99, currency=None, name="ASUS ROG")
        answer = answer_shopping_question("what is more expensive", products)
        self.assertEqual(answer, "ASUS ROG has the higher displayed price: $2,149.99 vs $899.00.")

    def test_tied_budget_prices_have_no_winner(self):
        self.products[1].update(price=50, price_display="$50")
        answer = answer_shopping_question("Budget $60", self.products)
        self.assertIn("tied", answer)
        self.assertNotIn("lower returned price", answer)

    def test_latest_user_follow_up_ignores_assistant_and_explicit_new_question(self):
        history = [{"role": "user", "content": "Under $100"}, {"role": "user", "content": "Under $60"},
                   {"role": "assistant", "content": "Under $5"}, {"role": "user", "content": "Which would you choose?"}]
        answer = answer_shopping_question("Which would you choose?", self.products, history=history)
        self.assertIn("60.00", answer)
        self.assertNotIn("100.00", answer)
        self.assertNotIn("5.00", answer)
        self.module.recommend_products.reset_mock()
        answer = answer_shopping_question("Which one is has higher customer ratings?", self.products, history=history)
        self.module.recommend_products.assert_not_called()
        self.assertIn("Trail Pack B", answer)
        self.assertIn("4.8/5 vs 4.1/5", answer)
        self.assertNotIn("60.00", answer)

    def test_spec_focus_and_priority_delegate_to_shopping_comparison(self):
        answer = answer_shopping_question("Which do you recommend?", self.products, "Size & weight")
        self.assertIn("Capacity", answer)
        self.assertIn("can't identify a supported winner", answer)
        self.module.comparison_rows.assert_called_once_with(self.products, confirmed=False)
        answer = answer_shopping_question("Which do you recommend?", self.products, "More customer reviews")
        self.module.recommend_products.assert_called_once_with(*self.products, "More customer reviews", confirmed=False)
        self.assertIn("Trail Pack B", answer)
        self.assertIn("96/100", answer)

    def test_no_winner_and_missing_rating(self):
        self.module.recommend_products.return_value = {"winner_index": None, "explanation": "Scores are tied."}
        self.assertIn("can't identify a clear winner", answer_shopping_question("Which is best?", self.products))
        self.products[1]["rating"] = 8
        answer = answer_shopping_question("Which is best?", self.products)
        self.assertIn("rating is missing or invalid", answer)
        self.assertNotIn("is the pick", answer)

    def test_unavailable_comparison_and_insufficient_input(self):
        with patch.dict(sys.modules, {"comparewise.shopping_comparison": None}):
            answer = answer_shopping_question("Which is cheaper?", self.products)
        self.assertIn("module is unavailable", answer)
        self.assertIn("exactly two", answer_shopping_question("Which?", []))
        self.assertIn("Tell me", answer_shopping_question("", self.products))

    def test_downside_does_not_invent_negatives(self):
        answer = answer_shopping_question("What are the downsides of Trail Pack B?", self.products)
        self.assertIn("no clear drawback", answer)
        self.assertNotIn("Trail Pack A:", answer)
        self.assertIn("won’t invent", answer)

    def test_weight_question_uses_weight_evidence_instead_of_recommending(self):
        answer = answer_shopping_question("Which one is heavier?", self.products)
        self.assertIn("weight not provided", answer)
        self.assertIn("Trail Pack A", answer)
        self.assertIn("Trail Pack B", answer)
        self.assertNotIn("comparison-based opinion", answer)

        self.products[0]["specifications"]["Weight"] = "500 g"
        self.products[1]["specifications"]["Weight"] = "1.2 kg"
        answer = answer_shopping_question("Which one is heavier?", self.products)
        self.assertIn("Trail Pack B is heavier", answer)
        self.assertIn("500 g", answer)
        self.assertIn("1.2 kg", answer)

    def test_eight_gaming_and_component_question_phrasings_use_matching_specs(self):
        products = [
            {"name": "Acer Nitro Intel Core 5 210H", "rating": 4.9,
             "specifications": {"GPU": "NVIDIA GeForce RTX 4060 Laptop GPU", "CPU": "Intel Core 5 210H",
                                "RAM": "16 GB", "Refresh rate": "144 Hz"}},
            {"name": "ASUS ROG Intel Core Ultra 9 285H", "rating": 4.2,
             "specifications": {"GPU": "NVIDIA GeForce RTX 4070 Laptop GPU", "CPU": "Intel Core Ultra 9 285H",
                                "RAM": "32 GB", "Refresh rate": "240 Hz"}},
        ]
        questions = (
            ("Which is just better for gaming?", "ASUS ROG", "listed graphics"),
            ("Which gaming laptop should I choose?", "ASUS ROG", "listed graphics"),
            ("Which has the better GPU?", "ASUS ROG", "listed graphics"),
            ("Which has better graphics?", "ASUS ROG", "listed graphics"),
            ("Which has the stronger processor?", "ASUS ROG", "listed processor"),
            ("Which has more RAM?", "ASUS ROG", "listed memory"),
            ("Which has a faster refresh rate?", "ASUS ROG", "listed refresh rate"),
            ("Which laptop is better for games?", "ASUS ROG", "listed graphics"),
        )
        for question, expected_name, expected_evidence in questions:
            with self.subTest(question=question):
                answer = answer_shopping_question(question, products)
                self.assertIn(expected_name, answer)
                self.assertIn(expected_evidence, answer)
                self.assertNotIn("balanced value", answer.casefold())

    def test_gaming_question_uses_listed_processor_when_gpu_is_missing(self):
        products = [
            {"name": "Acer Nitro Intel Core 5 210H", "rating": 4.9, "specifications": {}},
            {"name": "ASUS ROG Intel Core Ultra 9 285H", "rating": 4.2, "specifications": {}},
        ]
        answer = answer_shopping_question("Which is just better for gaming?", products)
        self.assertIn("ASUS ROG", answer)
        self.assertIn("listed processor", answer)

    def test_gaming_question_uses_gpu_in_title_and_flags_missing_comparison_gpu(self):
        products = [
            {"name": "Lenovo LOQ-E Gaming Laptop NVIDIA GeForce RTX 3050", "rating": 4.5,
             "specifications": {}},
            {"name": "Nimo 17.3 inch Gaming Laptop AMD Ryzen 5 6600H", "rating": 3.6,
             "specifications": {}},
        ]
        answer = answer_shopping_question("Which is better for laptop gaming?", products)
        self.assertIn("Lenovo LOQ-E", answer)
        self.assertIn("more promising gaming option", answer)
        self.assertIn("RTX 3050", answer)
        self.assertIn("GPU not listed", answer)
        self.assertIn("can't confirm head-to-head performance", answer)
        self.assertNotIn("customer rating", answer)

    def test_phone_storage_question_uses_capacity_in_product_title(self):
        products = [
            {"name": "Apple iPhone 18 Pro Max", "price": 1199, "currency": "USD", "rating": 4.7,
             "specifications": {}},
            {"name": "Google Pixel 11 Pro XL 512gb Olive Unlocked Android Smartphone", "price": 1099,
             "currency": "USD", "rating": 4.6, "specifications": {}},
        ]
        answer = answer_shopping_question("which is better for storage", products)
        self.assertIn("I'd tentatively recommend Google Pixel 11 Pro XL", answer)
        self.assertIn("512 GB", answer)
        self.assertIn("Apple iPhone 18 Pro Max's capacity isn't listed", answer)
        self.assertNotIn("Option 1", answer)
        self.assertNotIn("Option 2", answer)

    def test_what_do_you_think_followup_keeps_the_gaming_question_context(self):
        products = [
            {"name": "Acer Nitro Gaming Laptop", "specifications": {"GPU": "RTX 4060"}},
            {"name": "ASUS ROG Gaming Laptop", "specifications": {"GPU": "RTX 4070"}},
        ]
        history = [
            {"role": "user", "content": "Which is just better for gaming?"},
            {"role": "assistant", "content": "Previous response."},
            {"role": "user", "content": "like what do you think"},
        ]
        answer = answer_shopping_question("like what do you think", products, history=history)
        self.assertIn("ASUS ROG", answer)
        self.assertIn("listed graphics", answer)
        self.assertNotIn("couldn't find that detail", answer)

    def test_unrecognized_question_does_not_trigger_a_product_recommendation(self):
        answer = answer_shopping_question("Does it support satellite messaging?", self.products)
        self.assertIn("couldn't find that detail", answer)
        self.assertNotIn("is the pick", answer)

    def test_real_comparison_compatibility_and_confirmation(self):
        with patch.dict(sys.modules, {"comparewise.shopping_comparison": shopping_comparison}):
            compatible = deepcopy(self.products)
            compatible[0]["name"] = "Laptop A"
            compatible[1]["name"] = "Laptop B"
            self.assertIn("Laptop A", answer_shopping_question("Which is cheaper?", compatible))
            self.assertIn("can't identify a supported winner", answer_shopping_question("Which do you recommend?", compatible, "Size & weight"))
            uncertain = answer_shopping_question("Under $60", self.products)
            self.assertIn("confirmation", uncertain)
            confirmed = answer_shopping_question("Under $60", self.products, confirmed=True)
            self.assertIn("Trail Pack A is the only listing", confirmed)
            confirmed_recommendation = answer_shopping_question("Which is cheaper?", self.products, confirmed=True)
            self.assertIn("Trail Pack A", confirmed_recommendation)
            incompatible = deepcopy(compatible)
            incompatible[1]["name"] = "Smartphone B"
            answer = answer_shopping_question("Under $60", incompatible, confirmed=True)
            self.assertIn("cannot be compared", answer)
            self.assertNotIn("only listing", answer)


if __name__ == "__main__":
    unittest.main()