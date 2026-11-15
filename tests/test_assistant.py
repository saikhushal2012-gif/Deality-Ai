import unittest

from comparewise.assistant import answer_question
from comparewise.schemas import Product


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.products = [
            Product("Bundle One", "Computers", 900, performance=70, quality=75,
                    rating=4.2, features_score=72,
                    included_items=["monitor", "keyboard", "mouse"],
                    specifications={"Memory": "16 GB"}),
            Product("Tower Two", "Computers", 1100, performance=92, quality=86,
                    rating=4.6, features_score=90,
                    specifications={"Memory": "32 GB"}),
        ]

    def test_follow_up_uses_current_product_context(self):
        answer = answer_question("What are its disadvantages?", self.products,
                                 context_product="Tower Two")
        self.assertIn("Tower Two", answer)

    def test_missing_specs_are_not_invented(self):
        answer = answer_question("Which has better specs?", self.products)
        self.assertIn("no overall spec winner is assumed", answer)

    def test_accessory_what_if_uses_owned_items(self):
        answer = answer_question("What if I already own the keyboard?", self.products,
                                 setup_needs=["monitor", "keyboard", "mouse"])
        self.assertIn("own keyboard", answer)
        self.assertIn("best overall becomes", answer)

    def test_priority_what_if_is_recalculated(self):
        answer = answer_question("Choose the cheapest instead", self.products,
                                 priorities={"price": 0.2, "performance": 0.8})
        self.assertIn("recalculated ranking", answer)
        self.assertIn("lowest price prioritized", answer)


if __name__ == "__main__":
    unittest.main()
