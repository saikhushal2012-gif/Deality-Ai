import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import requests

from comparewise import gemini_assistant as assistant


def response(text="Option 1 is the better fit.", status=200):
    result = Mock(status_code=status)
    result.json.return_value = {
        "candidates": [{
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [{"text": text}]},
        }],
    }
    return result


class GeminiAssistantTests(unittest.TestCase):
    def setUp(self):
        self.products = [
            {"name": "Gaming Laptop A", "url": "https://shop.example/private", "image_url": "https://img.example/a",
             "price": 899, "price_display": "$899", "currency": "USD", "rating": 4.5,
             "review_count": 100, "seller": "Store A", "features": ["144 Hz display"],
             "specifications": {"GPU": "RTX 4060", "RAM": "16 GB"}, "description": "Gaming notebook",
             "private": "PRIVATE-EXTRA"},
            {"name": "Gaming Laptop B", "price": 1099, "price_display": "$1,099", "currency": "USD",
             "rating": 4.7, "review_count": 250, "seller": "Store B", "features": [],
             "specifications": {"GPU": "RTX 4070", "RAM": "16 GB"}},
        ]
        self.env = patch.dict("os.environ", {
            "GEMINI_API_KEY": "test-gemini-key", "GEMINI_MODEL": "gemini-3.8-flash",
            "SERPAPI_API_KEY": "test-serp-key",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.post_patch = patch.object(assistant.requests, "post", return_value=response())
        self.post = self.post_patch.start()
        self.addCleanup(self.post_patch.stop)

    def request_data(self):
        return json.loads(self.post.call_args.kwargs["json"]["contents"][-1]["parts"][0]["text"])

    def test_request_uses_gemini_api_and_bounded_listing_data(self):
        original = deepcopy(self.products)
        answer = assistant.request_model_answer(
            "Which gaming laptop is better?", self.products, "Balanced value",
            [{"role": "user", "content": "I mostly play new games"},
             {"role": "assistant", "content": "Compare GPU performance."}],
        )
        self.assertEqual(answer, "Option 1 is the better fit.")
        self.assertEqual(original, self.products)
        args, kwargs = self.post.call_args
        self.assertEqual(args, (assistant.API_URL.format(model="gemini-3.8-flash"),))
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "test-gemini-key")
        self.assertEqual(kwargs["timeout"], assistant.REQUEST_TIMEOUT)
        self.assertFalse(kwargs["allow_redirects"])
        body = kwargs["json"]
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertEqual(body["contents"][1]["role"], "model")
        self.assertEqual(self.request_data()["listings"][1]["specifications"]["GPU"], "RTX 4070")
        serialized = json.dumps(body)
        for forbidden in ("shop.example", "img.example", "PRIVATE-EXTRA", "test-gemini-key", "test-serp-key"):
            self.assertNotIn(forbidden, serialized)

    def test_key_is_required_and_incompatible_products_are_blocked(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            with self.assertRaises(assistant.ModelAssistantError):
                assistant.request_model_answer("Which is better?", self.products)
        incompatible = [self.products[0], {"name": "Smartphone B"}]
        with self.assertRaisesRegex(ValueError, "cannot be compared"):
            assistant.request_model_answer("Which is better?", incompatible, confirmed=True)
        self.post.assert_not_called()

    def test_provider_errors_are_safe_and_responses_are_validated(self):
        self.post.return_value = response(status=403)
        with self.assertRaisesRegex(assistant.ModelAssistantError, "rejected the key or request") as caught:
            assistant.request_model_answer("Which is better?", self.products)
        self.assertNotIn("test-gemini-key", str(caught.exception))
        self.post.return_value = response(status=429)
        with self.assertRaisesRegex(assistant.ModelAssistantError, "free-tier rate limit or quota"):
            assistant.request_model_answer("Which is better?", self.products)
        self.post.return_value = response(text=" ")
        with self.assertRaisesRegex(assistant.ModelAssistantError, "incomplete answer"):
            assistant.request_model_answer("Which is better?", self.products)
        self.post.side_effect = requests.Timeout("test-gemini-key")
        with self.assertRaisesRegex(assistant.ModelAssistantError, "timed out") as caught:
            assistant.request_model_answer("Which is better?", self.products)
        self.assertNotIn("test-gemini-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()