import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from comparewise.local_model_assistant import ModelAssistantError
from comparewise.sources import SerpApiError

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def widget(items, label):
    return next(item for item in items if item.label == label)


def product(identifier, name, seller, url, **fields):
    return {
        "id": identifier, "name": name, "seller": seller, "url": url,
        "image_url": "", "price": 50, "price_display": "US$50", "currency": "USD",
        "rating": 4.5, "review_count": 25, "features": ["Quartz movement"],
        "specifications": {"Material": "Steel"}, "description": "", "category": "",
        "source_note": "Unverified search result", **fields,
    }


WATCH_A = product("a", "Quartz Watch A", "Store A", "https://store-a.example/watch")
WATCH_B = product("b", "Automatic Watch B", "Store B", "https://store-b.example/watch",
                  price=90, price_display="US$90", rating=4.8, features=["Automatic movement"])
WATCH_D = product("e", "Quartz Watch D", "Store D", "https://store-d.example/watch",
                  price=65, price_display="US$65", rating=4.6)
WATCH_C = product("c", "Watch C", "Store C", "", image_url="javascript:alert(1)",
                  price=None, price_display="", rating=None, review_count=None, features=[], specifications={})
LAPTOP = product("l", "Laptop One", "Computer Shop", "https://computers.example/laptop",
                 specifications={"RAM": "16 GB"})
LAPTOP_TWO = product("l2", "Laptop Two", "Another Computer Shop", "https://computers.example/laptop-two",
                     price=65, price_display="US$65", specifications={"RAM": "16 GB", "CPU": "Ryzen 5"})
PHONE_A = product("p1", "Google Pixel 8a", "Phone Shop A", "https://phone-a.example/pixel", category="Smartphone")
PHONE_B = product("p2", "Samsung Galaxy A55", "Phone Shop B", "https://phone-b.example/galaxy", category="Smartphone")
DESKTOP = product("d", "Desktop Computer", "Other Shop", "https://other.example/desktop",
                  specifications={"RAM": "32 GB"})
MYSTERY = product("m", "Mystery Item", "Shop M", "https://m.example/item")
MYSTERY_2 = product("n", "Mystery Item Two", "Shop N", "https://n.example/item")


class ShoppingAppFlowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {"GEMINI_API_KEY": ""}))
        self.search_patch = patch("comparewise.sources.search_shopping_products")
        self.search = self.search_patch.start()
        self.addCleanup(self.search_patch.stop)
        self.search.side_effect = lambda query: deepcopy({
            "first watch": [WATCH_A, WATCH_C], "second watch": [WATCH_B, WATCH_C],
            "third watch": [WATCH_D],
            "laptop": [LAPTOP], "desktop": [DESKTOP],
            "first laptop": [LAPTOP], "second laptop": [LAPTOP_TWO],
            "pixel phone": [PHONE_A], "galaxy phone": [PHONE_B],
            "mystery": [MYSTERY, MYSTERY_2], "same": [WATCH_A],
        }.get(query, []))
        self.model_patch = patch("comparewise.gemini_assistant.request_model_answer",
                     return_value="Laptop Two is the better gaming pick based on its listed specs.")
        self.model = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertFalse(self.at.exception)
        self.search.assert_not_called()
        self.click("Start comparing →")

    def click(self, label):
        widget(self.at.button, label).click().run(timeout=30)
        self.assertFalse(self.at.exception)

    def query(self, number, value):
        widget(self.at.text_input, f"Search query {number}").set_value(value)
        self.click(f"Search {number}")

    def select(self, label, value):
        widget(self.at.selectbox, label).set_value(value).run(timeout=30)
        self.assertFalse(self.at.exception)

    def ask(self, question):
        widget(self.at.text_input, "Your question").set_value(question)
        self.click("Send")

    def test_home_intake_saves_shopping_goal_and_recommendation_priority(self):
        intake_app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertFalse(intake_app.exception)
        widget(intake_app.selectbox, "What are you shopping for?").set_value("Laptop for gaming")
        widget(intake_app.text_input, "Budget (optional)").set_value("$1,200")
        widget(intake_app.selectbox, "What matters most?").set_value("Performance & components")
        widget(intake_app.button, "Set my shopping goal").click().run(timeout=30)
        self.assertFalse(intake_app.exception)
        self.assertEqual(intake_app.session_state["shopping_goal"], "Laptop for gaming")
        self.assertEqual(intake_app.session_state["shopping_budget"], "$1,200")
        self.assertEqual(intake_app.session_state["shopping_priority"], "Performance & components")
        self.assertIn("Add at least two listings", intake_app.session_state["shopping_chat"][-1]["content"])

    def test_prelisting_question_uses_gemini_shopping_guidance(self):
        guidance_app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertFalse(guidance_app.exception)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
            with patch("comparewise.gemini_assistant.request_shopping_guidance",
                       return_value="Check GPU, RAM, and battery life.") as guidance:
                widget(guidance_app.text_input, "Your question").set_value("What should I look for?")
                widget(guidance_app.button, "Send").click().run(timeout=30)
        self.assertFalse(guidance_app.exception)
        self.assertIn("Check GPU", guidance_app.session_state["shopping_chat"][-1]["content"])
        guidance.assert_called_once()
        self.assertEqual(guidance.call_args.args[0], "What should I look for?")
        self.assertEqual(guidance.call_args.kwargs["api_key"], "test-gemini-key")

    def load_watches(self):
        self.query(1, "first watch")
        self.query(2, "second watch")
        self.click("Compare selected products →")

    def messages(self):
        return " ".join(str(item.value) for item in self.at.markdown)

    def test_independent_watch_searches_and_priority_without_refetching(self):
        self.assertEqual(self.search.call_count, 0)
        self.load_watches()
        self.assertEqual([call.args[0] for call in self.search.call_args_list], ["first watch", "second watch"])
        self.assertEqual([p["seller"] for p in self.at.session_state["comparison"]["products"]], ["Store A", "Store B"])
        self.assertIn("Quartz Watch A", self.messages())
        self.assertEqual(len(self.at.dataframe), 1)
        self.assertIn("Material", self.at.dataframe[0].value["Detail"].tolist())
        self.select("Recommendation priority", "Higher customer rating")
        self.assertIn("Automatic Watch B", self.messages())
        self.select("Recommendation priority", "Materials & build")
        self.assertIn("Recommended pick: Automatic Watch B", self.messages())
        self.assertEqual(self.search.call_count, 2)
        self.model.assert_not_called()

    def test_computers_and_missing_links_are_retailer_independent(self):
        self.query(1, "laptop")
        self.query(2, "desktop")
        self.click("Compare selected products →")
        self.assertEqual([p["name"] for p in self.at.session_state["comparison"]["products"]], ["Laptop One", "Desktop Computer"])
        self.assertIn("RAM", self.at.dataframe[0].value["Detail"].tolist())
        self.query(1, "first watch")
        self.select("Product option 1", 1)
        self.query(2, "second watch")
        self.click("Compare selected products →")
        self.assertTrue(any("Retailer link unavailable" in item.value for item in self.at.caption))
        self.assertIn("Image unavailable", self.messages())
        self.assertIn("Price unavailable", self.messages())

    def test_comparison_has_no_legacy_classifier_or_provider_selection(self):
        self.load_watches()
        self.assertFalse(any("Deality Match" in item.value for item in self.at.subheader))
        self.assertFalse(any(item.label == "AI assistant" for item in self.at.selectbox))
        self.assertTrue(any(item.label == "Your question" for item in self.at.text_input))

    def test_three_products_can_be_searched_compared_and_recommended(self):
        self.select("Products to compare", 3)
        self.query(1, "first watch")
        self.query(2, "second watch")
        self.query(3, "third watch")
        self.click("Compare selected products →")
        self.assertFalse(self.at.exception)
        self.assertEqual(len(self.at.session_state["comparison"]["products"]), 3)
        self.assertIn("Quartz Watch A", self.messages())
        self.assertEqual(self.at.dataframe[-1].value["Option 3"].tolist()[0], "Quartz Watch D")
        self.assertEqual(self.search.call_count, 3)

    def test_manual_listing_can_be_compared_without_fetching_its_url(self):
        widget(self.at.radio, "Input for option 1").set_value("Manual details").run(timeout=30)
        self.assertFalse(self.at.exception)
        for label, value in (("Product name 1", "Manual Watch"), ("Category or product type 1", "watch"),
                             ("Retailer 1 (optional)", "Local Store"),
                             ("Product page URL 1 (optional; never fetched)", "https://manual.example/watch"),
                             ("Price 1 (optional; numbers only)", "55"),
                             ("Currency code 1 (optional, e.g. USD)", "USD"),
                             ("Rating 1 out of 5 (optional)", "4.2"),
                             ("Specifications 1 (one Name: value per line)", "Battery: 2 days")):
            widget(self.at.text_input if label != "Specifications 1 (one Name: value per line)" else self.at.text_area, label).set_value(value)
        widget(self.at.button, "Save option 1").click().run(timeout=30)
        self.assertFalse(self.at.exception)
        self.query(2, "second watch")
        self.click("Compare selected products →")
        products = self.at.session_state["comparison"]["products"]
        self.assertEqual(products[0]["name"], "Manual Watch")
        self.assertIn("Manually entered", products[0]["source_note"])
        self.assertEqual(products[0]["url"], "https://manual.example/watch")
        self.assertEqual(self.search.call_count, 1)

    def test_incompatible_products_are_blocked(self):
        self.query(1, "first watch")
        self.query(2, "laptop")
        self.assertIn("cannot be compared", self.messages())
        self.click("Compare selected products →")
        self.assertIsNone(self.at.session_state["comparison"])
        self.assertFalse(self.at.checkbox)
        self.model.assert_not_called()

    def test_uncertain_confirmation_is_pair_specific_and_revocable(self):
        self.query(1, "mystery")
        self.query(2, "second watch")
        self.click("Compare selected products →")
        self.assertIsNone(self.at.session_state["comparison"])
        confirm = widget(self.at.checkbox, "I confirm these two selected products serve the same purpose")
        confirm.check().run(timeout=30)
        self.click("Compare selected products →")
        self.assertTrue(self.at.session_state["comparison"]["confirmed"])
        self.ask("Which is cheaper?")
        self.assertEqual(len(self.at.session_state["shopping_chat"]), 2)
        self.select("Product option 1", 1)
        self.assertIsNone(self.at.session_state["comparison"])
        self.assertEqual(self.at.session_state["shopping_chat"], [])
        self.assertFalse(widget(self.at.checkbox, "I confirm these two selected products serve the same purpose").value)
        self.select("Product option 1", 0)
        self.assertFalse(widget(self.at.checkbox, "I confirm these two selected products serve the same purpose").value)
        widget(self.at.checkbox, "I confirm these two selected products serve the same purpose").check().run(timeout=30)
        self.click("Compare selected products →")
        widget(self.at.checkbox, "I confirm these two selected products serve the same purpose").uncheck().run(timeout=30)
        self.assertIsNone(self.at.session_state["comparison"])

    def test_duplicate_rejected_even_across_queries_and_cache_used(self):
        self.query(1, "same")
        self.query(2, "same")
        self.assertEqual(self.search.call_count, 1)
        self.click("Compare selected products →")
        self.assertIsNone(self.at.session_state["comparison"])
        self.assertIn("same listing", self.messages())

    def test_selection_invalidates_comparison_and_history_before_chat(self):
        self.load_watches()
        self.ask("Which has automatic movement?")
        self.assertIn("Quartz Watch A", self.at.session_state["shopping_chat"][-1]["content"])
        self.assertIn("no matching detail was returned", self.at.session_state["shopping_chat"][-1]["content"])
        self.select("Product option 2", 1)
        self.assertIsNone(self.at.session_state["comparison"])
        self.assertEqual(self.at.session_state["shopping_chat"], [])
        self.ask("Which is cheaper?")
        self.assertIn("Compare the selected products first", self.messages())
        self.assertEqual(self.at.session_state["shopping_chat"], [])
        self.assertEqual(self.search.call_count, 2)
        self.click("Compare selected products →")
        self.assertEqual(self.at.session_state["shopping_chat"], [])
        self.click("Clear searches, comparison and chat")
        self.assertIsNone(self.at.session_state["comparison"])
        self.assertEqual(self.at.session_state["shopping_chat"], [])
        self.assertEqual(self.at.session_state["shopping_cache"], {})

    def test_assistant_answers_confirmed_pair_without_external_model(self):
        self.query(1, "mystery")
        self.query(2, "second watch")
        widget(self.at.checkbox, "I confirm these two selected products serve the same purpose").check().run(timeout=30)
        self.click("Compare selected products →")
        self.ask("Which is cheaper?")
        self.model.assert_not_called()
        self.assertIn("Mystery Item", self.at.session_state["shopping_chat"][-1]["content"])
        self.assertEqual(self.search.call_count, 2)

    def test_recommendation_is_immediate_and_chat_uses_rules(self):
        first = deepcopy(WATCH_A)
        first.update(price=899, price_display="$899.00", currency=None)
        second = deepcopy(WATCH_B)
        second.update(price=1399, price_display="$1,399.00", currency=None)
        self.search.side_effect = lambda query: deepcopy({
            "first watch": [first], "second watch": [second],
        }.get(query, []))
        self.load_watches()
        self.assertIn("DEALITY AI · BALANCED VALUE", self.messages())
        self.assertIn("Quartz Watch A", self.messages())
        self.ask("Which is cheaper?")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("Quartz Watch A has the lower displayed price", answer)
        self.assertIn("$899.00 vs $1,399.00", answer)
        self.assertNotIn("currency", answer)
        self.assertNotIn("balanced value", answer)
        self.ask("what is more expensive")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("Automatic Watch B", answer)
        self.assertIn("has the higher displayed price", answer)
        self.assertIn("$1,399.00 vs $899.00", answer)
        urls = [item.proto.url for item in self.at.get("link_button")]
        self.assertGreaterEqual(urls.count(WATCH_A["url"]), 2)
        self.assertGreaterEqual(urls.count(WATCH_B["url"]), 2)
        self.model.assert_not_called()
        self.assertEqual(self.search.call_count, 2)

    def test_missing_weight_answer_offers_both_product_listing_links(self):
        self.load_watches()
        self.ask("Which one is heavier?")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("weight not provided", answer)
        self.assertNotIn("comparison-based opinion", answer)
        urls = [item.proto.url for item in self.at.get("link_button")]
        self.assertGreaterEqual(urls.count(WATCH_A["url"]), 2)
        self.assertGreaterEqual(urls.count(WATCH_B["url"]), 2)

    def test_comparison_page_has_no_external_assistant_selector(self):
        self.load_watches()
        self.assertFalse(any(item.label == "AI assistant" for item in self.at.selectbox))
        self.assertTrue(any(item.label == "Your question" for item in self.at.text_input))

    def test_gemini_answers_open_ended_gaming_question_when_configured(self):
        self.query(1, "first laptop")
        self.query(2, "second laptop")
        self.click("Compare selected products →")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
            self.ask("Which is the better gaming laptop?")
        self.assertIn("Laptop Two is the better gaming pick", self.at.session_state["shopping_chat"][-1]["content"])
        self.model.assert_called_once()
        self.assertEqual(self.model.call_args.args[0], "Which is the better gaming laptop?")
        self.assertEqual(self.model.call_args.kwargs["api_key"], "test-gemini-key")

    def test_gaming_question_uses_hardware_evidence_without_gemini_key(self):
        first = deepcopy(LAPTOP)
        first.update(name="Acer Nitro Gaming Laptop Intel Core 5 210H", specifications={
            "CPU": "Intel Core 5 210H", "GPU": "NVIDIA GeForce RTX 4060 Laptop GPU", "RAM": "16 GB",
        })
        second = deepcopy(LAPTOP_TWO)
        second.update(name="ASUS ROG Gaming Laptop Intel Core Ultra 9 285H", specifications={
            "CPU": "Intel Core Ultra 9 285H", "GPU": "NVIDIA GeForce RTX 4070 Laptop GPU", "RAM": "32 GB",
        })
        self.search.side_effect = lambda query: deepcopy({
            "acer": [first], "asus": [second],
        }.get(query, []))
        self.query(1, "acer")
        self.query(2, "asus")
        self.click("Compare selected products →")
        self.ask("Which is just better for gaming?")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("ASUS ROG", answer)
        self.assertIn("listed graphics", answer)
        self.model.assert_not_called()

    def test_short_followups_keep_gaming_context(self):
        first = deepcopy(LAPTOP)
        first.update(name="Acer Nitro Gaming Laptop", specifications={"GPU": "RTX 4060"})
        second = deepcopy(LAPTOP_TWO)
        second.update(name="ASUS ROG Gaming Laptop", specifications={"GPU": "RTX 4070"})
        self.search.side_effect = lambda query: deepcopy({
            "acer": [first], "asus": [second],
        }.get(query, []))
        self.query(1, "acer")
        self.query(2, "asus")
        self.click("Compare selected products →")
        self.ask("Which is just better for gaming?")
        self.ask("like what do you think")
        self.ask("why")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("ASUS ROG", answer)
        self.assertIn("listed graphics", answer)
        self.assertNotIn("couldn't find that detail", answer)
        self.model.assert_not_called()

    def test_gemini_failure_falls_back_to_listing_rules(self):
        self.load_watches()
        self.model.side_effect = ModelAssistantError("provider unavailable")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
            self.ask("Which is the better option for everyday use?")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertTrue(answer)
        self.assertIn("Quartz Watch A", answer)
        self.model.assert_called_once()

    def test_direct_price_question_stays_concise_with_gemini_configured(self):
        self.load_watches()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
            self.ask("what is more expensive")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("Automatic Watch B is more expensive", answer)
        self.assertIn("US$90 vs US$50", answer)
        self.model.assert_not_called()

    def test_customer_rating_question_uses_listing_values_with_gemini_configured(self):
        self.load_watches()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
            self.ask("Which one is has higher customer ratings?")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("Automatic Watch B", answer)
        self.assertIn("4.8/5 vs 4.5/5", answer)
        self.model.assert_not_called()

    def test_phone_storage_question_uses_named_listing_evidence_with_gemini_configured(self):
        iphone = product("i", "Apple iPhone 18 Pro Max", "Apple", "https://apple.example/iphone",
                         category="Smartphone", price=1199, rating=4.7, specifications={})
        pixel = product("p", "Google Pixel 11 Pro XL 512gb Olive Unlocked Android Smartphone",
                        "Google", "https://google.example/pixel", category="Smartphone",
                        price=1099, rating=4.6, specifications={})
        self.search.side_effect = lambda query: {"iphone": [iphone], "pixel": [pixel]}.get(query, [])
        self.query(1, "iphone")
        self.query(2, "pixel")
        self.click("Compare selected products →")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"}):
            self.ask("which is better for storage")
        answer = self.at.session_state["shopping_chat"][-1]["content"]
        self.assertIn("I'd tentatively recommend Google Pixel 11 Pro XL", answer)
        self.assertIn("512 GB", answer)
        self.assertNotIn("Option 1", answer)
        self.model.assert_not_called()

    def test_search_error_clears_previous_comparison_without_leaking(self):
        self.load_watches()
        self.search.side_effect = SerpApiError("Search quota reached")
        self.query(1, "new search")
        self.assertIn("Search quota reached", self.messages())
        self.assertIsNone(self.at.session_state["comparison"])
        self.assertEqual(self.at.session_state["results_1"], [])

    def test_escaped_image_and_unavailable_retailer_redirect(self):
        unsafe = product("z", '<Watch "Z">', "Shop", "https://google.com/redirect",
                         image_url='https://images.example/image?x="quoted"&y=1')
        self.search.side_effect = lambda query: deepcopy([unsafe] if query == "unsafe" else [WATCH_B])
        self.query(1, "unsafe")
        self.query(2, "other")
        self.click("Compare selected products →")
        self.assertIn("&lt;Watch", self.messages())
        self.assertIn("&quot;quoted&quot;&amp;y=1", self.messages())
        self.assertTrue(any("Retailer link unavailable" in item.value for item in self.at.caption))


if __name__ == "__main__":
    unittest.main()
