"""Exercise the UI through real Shopping normalization, without network access.

These are representative SerpApi google_shopping response shapes, not live
offers. All merchant and image URLs are deliberately fictional. The optional
specifications fields are fixture data, not assumed to be present in results.
"""

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, call, patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
API_KEY = "test-only-key"

PAYLOADS = {
    "quartz wrist watch": {
        "search_metadata": {"status": "Success"},
        "search_parameters": {"engine": "google_shopping", "q": "quartz wrist watch"},
        "shopping_results": [
            {"position": 1, "title": "Quartz Watch Classic", "source": "Other Shop",
             "link": "https://other.example/watch", "price": "US$120.00", "extracted_price": 120.0},
            {"position": 2, "title": "Quartz Wrist Watch A", "source": "Amazon",
             "link": "https://amazon.example/watch-a", "price": "US$49.95", "extracted_price": 49.95,
             "rating": 4.4, "reviews": 1234, "thumbnail": "https://images.example/watch-a.jpg",
             "extensions": ["Free delivery"], "specifications": {"Movement": "Quartz", "Material": "Steel"}},
        ],
    },
    "automatic wrist watch": {
        "search_metadata": {"status": "Success"},
        "search_parameters": {"engine": "google_shopping", "q": "automatic wrist watch"},
        "shopping_results": [
            {"position": 1, "title": "Automatic Wrist Watch B", "source": "Walmart",
             "product_link": "https://walmart.example/watch-b", "price": "USD 89.00", "extracted_price": 89.0,
             "rating": 4.8, "reviews": "2,345", "thumbnail": "https://images.example/watch-b.jpg",
             "specifications": {"movement": "Automatic", "Water resistance": "50 m"}},
        ],
    },
    "ultrabook laptop": {
        "search_metadata": {"status": "Success"},
        "search_parameters": {"engine": "google_shopping", "q": "ultrabook laptop"},
        "shopping_results": [
            {"position": 1, "title": "Ultrabook Laptop A", "source": "eBay",
             "link": "https://ebay.example/laptop-a", "price": "US$599.00", "extracted_price": 599.0,
             "rating": 4.3, "reviews": 310, "specifications": {"RAM": "16 GB", "Storage": "512 GB"}},
        ],
    },
    "desktop computer": {
        "search_metadata": {"status": "Success"},
        "search_parameters": {"engine": "google_shopping", "q": "desktop computer"},
        "shopping_results": [
            {"position": 1, "title": "Desktop Computer B", "source": "Best Buy",
             "link": "https://bestbuy.example/desktop-b", "price": "US$799.00", "extracted_price": 799.0,
             "rating": 4.7, "reviews": 590, "specifications": {"ram": "32 GB", "Processor": "Example CPU"}},
        ],
    },
}


def widget(items, label):
    return next(item for item in items if item.label == label)


class ShoppingEndToEndTests(unittest.TestCase):
    def setUp(self):
        # Do not let an actual .env, environment key, or external HTTP request
        # affect the test. Patches remain active across Streamlit reruns.
        self.enterContext(patch.dict("os.environ", {
            "SERPAPI_API_KEY": API_KEY, "SERPAPI_KEY": "", "SERAPI_KEY": "",
            "GEMINI_API_KEY": "",
        }))
        self.enterContext(patch("dotenv.load_dotenv", return_value=False))
        self.enterContext(patch("comparewise.sources.dotenv_values", return_value={}))
        self.get = self.enterContext(patch("comparewise.sources.requests.get", side_effect=self.shopping_response))
        self.enterContext(patch("requests.post", side_effect=AssertionError("Unexpected HTTP POST")))
        self.at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assert_no_exceptions()
        self.get.assert_not_called()
        self.click("Start comparing →")

    def shopping_response(self, url, *, params, timeout, allow_redirects):
        # Unknown requests fail immediately rather than falling through to the network.
        self.assertEqual(url, "https://serpapi.com/search.json")
        self.assertEqual(params["engine"], "google_shopping")
        self.assertEqual(params["api_key"], API_KEY)
        self.assertEqual(set(params), {"engine", "q", "api_key"})
        self.assertEqual(timeout, (5, 30))
        self.assertFalse(allow_redirects)
        self.assertIn(params["q"], PAYLOADS)
        response = Mock(status_code=200)
        response.json.return_value = deepcopy(PAYLOADS[params["q"]])
        return response

    def assert_no_exceptions(self):
        self.assertFalse(self.at.exception, [str(error) for error in self.at.exception])

    def click(self, label):
        widget(self.at.button, label).click().run(timeout=30)
        self.assert_no_exceptions()

    def select(self, label, value):
        widget(self.at.selectbox, label).set_value(value).run(timeout=30)
        self.assert_no_exceptions()

    def search(self, number, query):
        widget(self.at.text_input, f"Search query {number}").set_value(query)
        self.click(f"Search {number}")

    def assert_requests(self, *queries):
        self.assertEqual(self.get.call_args_list, [
            call("https://serpapi.com/search.json",
                 params={"engine": "google_shopping", "q": query, "api_key": API_KEY},
                 timeout=(5, 30), allow_redirects=False)
            for query in queries
        ])

    def recommendation(self):
        return next(item.value for item in self.at.markdown if 'class="recommendation"' in item.value)

    def assert_listings(self, expected):
        products = self.at.session_state["comparison"]["products"]
        for product, (name, seller, price, display, url) in zip(products, expected):
            self.assertEqual((product["name"], product["seller"], product["price"],
                              product["price_display"], product["currency"], product["url"]),
                             (name, seller, price, display, "USD", url))
        self.assertEqual(len(products), 2)
        self.assertEqual([link.proto.url for link in self.at.get("link_button")],
                         [listing[4] for listing in expected])
        self.assertEqual([caption.value for caption in self.at.caption if caption.value.startswith("Seller:")],
                         [f"Seller: {listing[1]}" for listing in expected])
        rows = self.at.dataframe[-1].value.set_index("Detail")
        self.assertEqual(rows.loc["Seller"].tolist(), [listing[1] for listing in expected])
        self.assertEqual(rows.loc["Price"].tolist(), [listing[3] for listing in expected])
        self.assertEqual(rows.loc["URL"].tolist(), [listing[4] for listing in expected])
        return rows

    def test_watch_search_selection_recommendation_and_specs(self):
        self.search(1, "quartz wrist watch")
        self.search(2, "automatic wrist watch")
        self.select("Product option 1", 1)
        self.select("Recommendation priority", "Lower price")
        self.click("Compare selected products →")
        rows = self.assert_listings([
            ("Quartz Wrist Watch A", "Amazon", 49.95, "US$49.95", "https://amazon.example/watch-a"),
            ("Automatic Wrist Watch B", "Walmart", 89.0, "USD 89.00", "https://walmart.example/watch-b"),
        ])
        self.assertIn("<h3>Quartz Wrist Watch A</h3>", self.recommendation())
        self.assertEqual(rows.loc["Movement"].tolist(), ["Quartz", "Automatic"])
        self.assertEqual(rows.loc["Material"].tolist(), ["Steel", "Not listed"])
        self.assertEqual(rows.loc["Water resistance"].tolist(), ["Not listed", "50 m"])
        self.assertEqual(rows.loc["Features"].tolist(), ["Free delivery", "Not listed"])
        self.select("Recommendation priority", "Higher customer rating")
        self.assertIn("<h3>Automatic Wrist Watch B</h3>", self.recommendation())
        self.select("Recommendation priority", "Materials & build")
        self.assertIn("<h3>No winner established</h3>", self.recommendation())
        self.assertEqual(len(self.at.dataframe), 2)
        self.assertIn("Material", self.at.dataframe[0].value["Detail"].tolist())
        self.assert_requests("quartz wrist watch", "automatic wrist watch")

    def test_computer_search_recommendation_and_dynamic_specs(self):
        self.search(1, "ultrabook laptop")
        self.search(2, "desktop computer")
        self.select("Recommendation priority", "Lower price")
        self.click("Compare selected products →")
        rows = self.assert_listings([
            ("Ultrabook Laptop A", "eBay", 599.0, "US$599.00", "https://ebay.example/laptop-a"),
            ("Desktop Computer B", "Best Buy", 799.0, "US$799.00", "https://bestbuy.example/desktop-b"),
        ])
        self.assertIn("<h3>Ultrabook Laptop A</h3>", self.recommendation())
        self.assertEqual(rows.loc["RAM"].tolist(), ["16 GB", "32 GB"])
        self.assertEqual(rows.loc["Storage"].tolist(), ["512 GB", "Not listed"])
        self.assertEqual(rows.loc["Processor"].tolist(), ["Not listed", "Example CPU"])
        self.assertNotIn("Movement", rows.index)
        self.assert_requests("ultrabook laptop", "desktop computer")

    def test_watch_and_computer_mismatch_is_blocked(self):
        self.search(1, "quartz wrist watch")
        self.select("Product option 1", 1)
        self.search(2, "ultrabook laptop")
        self.assertTrue(any("cannot be compared" in item.value for item in self.at.markdown))
        self.assertFalse(self.at.checkbox)
        self.click("Compare selected products →")
        self.assertIsNone(self.at.session_state["comparison"])
        self.assertTrue(any("cannot be compared" in item.value for item in self.at.markdown))
        self.assertFalse(self.at.dataframe)
        self.assert_requests("quartz wrist watch", "ultrabook laptop")


if __name__ == "__main__":
    unittest.main()