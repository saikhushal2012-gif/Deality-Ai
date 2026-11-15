import unittest
from unittest.mock import Mock, patch

import requests

from comparewise.sources import SerpApiError, search_shopping_products


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


@patch.dict("os.environ", {"SERPAPI_API_KEY": "secret-key"}, clear=True)
class ShoppingSourcesTests(unittest.TestCase):
    @patch("comparewise.sources.requests.get")
    def test_request_and_multiple_retailers(self, get):
        get.return_value = response({"shopping_results": [
            {"title": "Watch", "link": "https://shop.example/watch", "source": "Shop", "price": "US$49.95", "extracted_price": 49.95, "rating": 4.5, "reviews": "1,234", "thumbnail": "https://img.example/watch.jpg", "snippet": "A watch", "features": ["Quartz"], "specifications": {"Case": "Steel"}, "category": "Watches"},
            {"title": "Watch", "product_link": "https://other.example/watch", "source": "Other", "price": "€55,50"},
        ]})
        products = search_shopping_products("  wrist\n watch  ")
        self.assertEqual(get.call_args.args[0], "https://serpapi.com/search.json")
        self.assertEqual(get.call_args.kwargs["params"], {"engine": "google_shopping", "q": "wrist watch", "api_key": "secret-key"})
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertEqual(len(products), 2)
        first, second = products
        self.assertEqual((first["name"], first["seller"], first["url"]), ("Watch", "Shop", "https://shop.example/watch"))
        self.assertEqual((first["price"], first["currency"], first["price_display"]), (49.95, "USD", "US$49.95"))
        self.assertEqual((first["rating"], first["review_count"]), (4.5, 1234))
        self.assertEqual(first["image_url"], "https://img.example/watch.jpg")
        self.assertEqual((first["description"], first["features"], first["specifications"], first["category"]), ("A watch", ["Quartz"], {"Case": "Steel"}, "Watches"))
        self.assertEqual((second["price"], second["currency"], second["seller"]), (55.5, "EUR", "Other"))
        self.assertNotEqual(first["id"], second["id"])
        self.assertIn("SerpApi", first["source_note"])

    @patch("comparewise.sources.requests.get")
    def test_missing_data_and_documented_google_product_link(self, get):
        get.return_value = response({"shopping_results": [None, {}, {"title": "Phone", "product_id": "123", "product_link": "https://www.google.com/search?ibp=oshop", "multiple_sources": True, "extensions": ["Free delivery"]}, {"title": "Other", "thumbnail": "javascript:alert(1)", "rating": 8, "reviews": -3}]})
        items = search_shopping_products("phone")
        self.assertEqual(len(items), 2)
        item = items[0]
        self.assertEqual((item["url"], item["seller"], item["image_url"], item["description"], item["category"], item["price_display"]), ("", "", "", "", "", ""))
        self.assertEqual((item["price"], item["currency"], item["rating"], item["review_count"]), (None, None, None, None))
        self.assertEqual((item["features"], item["specifications"]), (["Free delivery"], {}))
        self.assertEqual(item["shopping_url"], "https://www.google.com/search?ibp=oshop")
        self.assertIn("multiple sellers", item["source_note"])
        self.assertNotEqual(item["id"], "123")
        self.assertEqual((items[1]["image_url"], items[1]["rating"], items[1]["review_count"]), ("", None, None))

    @patch("comparewise.sources.requests.get")
    def test_redirects_and_unsafe_urls_are_not_followed(self, get):
        links = ["https://www.google.co.in/aclk?x=1", "https://serpapi.com/search.json?engine=x", "https://googleadservices.com/pagead/aclk", "https://bit.ly/abc", "https://shop.example/redirect?url=https%3A%2F%2Fother.example", "https://shop.example/item?redirect_url=https%3A%2F%2Fother.example", "javascript:alert(1)", "https://user:pass@shop.example/item", "https://shop.example:bad/item"]
        get.return_value = response({"shopping_results": [{"title": str(i), "link": link} for i, link in enumerate(links)]})
        products = search_shopping_products("item")
        self.assertTrue(all(not p["url"] for p in products))
        self.assertEqual(get.call_count, 1)

    @patch("comparewise.sources.requests.get")
    def test_deduplication_does_not_conflate_sellers_or_prices(self, get):
        a = {"title": "Item", "source": "Seller A", "product_id": "same", "price": "$10", "extracted_price": 10}
        b = {**a, "source": "Seller B"}
        c = {**a, "price": "$11", "extracted_price": 11}
        anonymous = {"title": "Anonymous"}
        get.return_value = response({"shopping_results": [a, a.copy(), b, c, anonymous, anonymous.copy()]})
        first = search_shopping_products("item")
        second = search_shopping_products("item")
        self.assertEqual(len(first), 5)
        self.assertEqual([p["id"] for p in first], [p["id"] for p in second])
        self.assertEqual(len({p["id"] for p in first}), 5)
        self.assertEqual([p["seller"] for p in first], ["Seller A", "Seller B", "Seller A", "", ""])
        self.assertEqual(len(search_shopping_products("item", limit=2)), 2)

    @patch("comparewise.sources.requests.get")
    def test_prices_do_not_invent_currency_or_misparse_ranges(self, get):
        examples = [
            ({"price": "$1,234.50", "extracted_price": 1234.5}, (1234.5, None)),
            ({"price": "CA$12.50"}, (12.5, "CAD")),
            ({"price": "A$12.50"}, (12.5, "AUD")),
            ({"price": "£1,234.50"}, (1234.5, "GBP")),
            ({"price": "1.234,50 EUR"}, (1234.5, "EUR")),
            ({"price": "₹100"}, (100.0, "INR")),
            ({"price": "¥100"}, (100.0, None)),
            ({"price": "$1,234"}, (None, None)),
            ({"price": "$10 - $20", "extracted_price": 10}, (None, None)),
            ({"price": "From $10", "extracted_price": 10}, (None, None)),
            ({"price": "$10/month", "extracted_price": 10}, (None, None)),
            ({"price": "EUR 1 2"}, (None, "EUR")),
            ({"extracted_price": 10, "currency": "USD"}, (10, "USD")),
            ({"price": "USD 10", "currency": "EUR"}, (10, None)),
            ({"extracted_price": float("nan")}, (None, None)),
        ]
        get.return_value = response({"shopping_results": [{"title": str(i), **hit} for i, (hit, _) in enumerate(examples)]})
        products = search_shopping_products("item")
        self.assertEqual([(p["price"], p["currency"]) for p in products], [expected for _, expected in examples])

    @patch("comparewise.sources.requests.get")
    def test_validation_and_empty_results(self, get):
        for value in ("", "  \n ", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "search phrase"):
                search_shopping_products(value)
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                search_shopping_products("item", value)
        get.assert_not_called()
        get.return_value = response({})
        self.assertEqual(search_shopping_products("item"), [])
        get.return_value = response({"shopping_results": {}})
        with self.assertRaisesRegex(SerpApiError, "format"):
            search_shopping_products("item")

    @patch("comparewise.sources.requests.get")
    def test_http_and_payload_errors_do_not_leak_secrets(self, get):
        for status in (301, 401, 402, 403, 429, 500):
            get.return_value = response({}, status)
            with self.subTest(status=status), self.assertRaises(SerpApiError) as caught:
                search_shopping_products("item")
            self.assertEqual(caught.exception.status_code, status)
            self.assertNotIn("secret-key", str(caught.exception))
        for payload in ({"error": "secret-key"}, {"search_metadata": {"status": "Error"}}, [], "bad"):
            get.return_value = response(payload)
            with self.subTest(payload=type(payload)), self.assertRaises(SerpApiError) as caught:
                search_shopping_products("item")
            self.assertNotIn("secret-key", str(caught.exception))
        get.return_value = response(None)
        get.return_value.json.side_effect = ValueError("secret-key")
        with self.assertRaisesRegex(SerpApiError, "unreadable"):
            search_shopping_products("item")
        get.side_effect = requests.Timeout("secret-key")
        with self.assertRaises(SerpApiError) as caught:
            search_shopping_products("item")
        self.assertNotIn("secret-key", str(caught.exception))


class CredentialTests(unittest.TestCase):
    @patch("comparewise.sources.requests.get")
    def test_all_environment_aliases(self, get):
        get.return_value = response({})
        for name in ("SERPAPI_API_KEY", "SERPAPI_KEY", "SERAPI_KEY"):
            with self.subTest(name=name), patch.dict("os.environ", {name: " key "}, clear=True):
                search_shopping_products("item")
                self.assertEqual(get.call_args.kwargs["params"]["api_key"], "key")

    @patch("comparewise.sources.dotenv_values")
    @patch("comparewise.sources.requests.get")
    @patch.dict("os.environ", {"SERPAPI_API_KEY": " "}, clear=True)
    def test_dotenv_fallback_and_missing_key(self, get, dotenv):
        get.return_value = response({})
        dotenv.return_value = {"SERPAPI_API_KEY": "dotenv-key"}
        search_shopping_products("item")
        self.assertEqual(get.call_args.kwargs["params"]["api_key"], "dotenv-key")
        dotenv.return_value = {}
        with self.assertRaisesRegex(SerpApiError, "SERAPI_KEY"):
            search_shopping_products("item")
        self.assertEqual(get.call_count, 1)


if __name__ == "__main__":
    unittest.main()