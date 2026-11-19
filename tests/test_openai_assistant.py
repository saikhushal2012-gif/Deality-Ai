import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import requests

from comparewise import openai_assistant as assistant


def response(text="Option 1 is the better fit.", status=200):
    result = Mock(status_code=status)
    result.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
    }
    return result


class OpenAIAssistantTests(unittest.TestCase):
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
            "OPENAI_API_KEY": "test-openai-key", "OPENAI_MODEL": "gpt-4o-mini",
            "SERPAPI_API_KEY": "test-serp-key",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.post_patch = patch.object(assistant.requests, "post", return_value=response())
        self.post = self.post_patch.start()
        self.addCleanup(self.post_patch.stop)

    def request_data(self):
        return json.loads(self.post.call_args.kwargs["json"]["messages"][1]["content"])

    def test_request_uses_bounded_listing_data_and_returns_model_answer(self):
        original = deepcopy(self.products)
        answer = assistant.request_model_answer(
            "Which gaming laptop is better?", self.products, "Balanced value",
            [{"role": "user", "content": "I mostly play new games"}],
        )
        self.assertEqual(answer, "Option 1 is the better fit.")
        self.assertEqual(original, self.products)
        args, kwargs = self.post.call_args
        self.assertEqual(args, (assistant.API_URL,))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-openai-key")
        self.assertEqual(kwargs["timeout"], assistant.REQUEST_TIMEOUT)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["json"]["model"], "gpt-4o-mini")
        self.assertFalse(kwargs["json"]["stream"])
        data = self.request_data()
        self.assertEqual(data["listings"][1]["specifications"]["GPU"], "RTX 4070")
        self.assertEqual(data["question"], "Which gaming laptop is better?")
        self.assertNotIn("shop.example", json.dumps(data))
        self.assertNotIn("img.example", json.dumps(data))
        self.assertNotIn("PRIVATE-EXTRA", json.dumps(data))
        self.assertNotIn("test-openai-key", json.dumps(kwargs["json"]))
        self.assertNotIn("test-serp-key", json.dumps(kwargs["json"]))

    def test_credentials_are_required_and_untrusted_mismatched_products_are_blocked(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with self.assertRaises(assistant.ModelAssistantError):
                assistant.request_model_answer("Which is better?", self.products)
        incompatible = [self.products[0], {"name": "Smartphone B"}]
        with self.assertRaisesRegex(ValueError, "cannot be compared"):
            assistant.request_model_answer("Which is better?", incompatible, confirmed=True)
        self.post.assert_not_called()

    def test_provider_errors_are_safe_and_responses_are_validated(self):
        self.post.return_value = response(status=401)
        with self.assertRaisesRegex(assistant.ModelAssistantError, "API key was rejected") as caught:
            assistant.request_model_answer("Which is better?", self.products)
        self.assertNotIn("test-openai-key", str(caught.exception))
        self.post.return_value = response(text=" ")
        with self.assertRaisesRegex(assistant.ModelAssistantError, "incomplete answer"):
            assistant.request_model_answer("Which is better?", self.products)
        self.post.side_effect = requests.Timeout("test-openai-key")
        with self.assertRaisesRegex(assistant.ModelAssistantError, "timed out") as caught:
            assistant.request_model_answer("Which is better?", self.products)
        self.assertNotIn("test-openai-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()