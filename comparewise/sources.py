"""Google Shopping results via SerpApi.

See https://serpapi.com/google-shopping-api. Search results are offers, not
verified product-detail records. No extra requests are made to follow links.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import requests
from dotenv import dotenv_values

SERPAPI_URL = "https://serpapi.com/search.json"
_KEY_NAMES = ("SERPAPI_API_KEY", "SERPAPI_KEY", "SERAPI_KEY")


class SerpApiError(ValueError):
    """A safe error which does not expose credentials or response content."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _serpapi_request(params: dict) -> dict:
    # Respect all process environment aliases before loading the local .env.
    api_key = next((os.environ[name].strip() for name in _KEY_NAMES if os.getenv(name, "").strip()), "")
    if not api_key:
        values = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
        api_key = next((values[name].strip() for name in _KEY_NAMES if values.get(name, "") and values[name].strip()), "")
    if not api_key:
        raise SerpApiError("Google Shopping is not configured. Set SERPAPI_API_KEY, SERPAPI_KEY, or SERAPI_KEY in the local .env file.")
    try:
        response = requests.get(
            SERPAPI_URL, params={**params, "api_key": api_key},
            timeout=(5, 30), allow_redirects=False,
        )
    except requests.RequestException:
        raise SerpApiError("SerpApi could not be reached. Check the connection and try again.") from None
    if not 200 <= response.status_code < 300:
        messages = {
            401: "SerpApi rejected the API credentials. Check the key saved in .env.",
            402: "SerpApi credits are exhausted (HTTP 402). Check your plan or quota.",
            403: "SerpApi denied this request. Check the account and API access.",
            429: "SerpApi rate limit or search quota reached. Try again later.",
        }
        raise SerpApiError(messages.get(response.status_code, f"SerpApi request failed (HTTP {response.status_code})."), response.status_code)
    try:
        payload = response.json()
    except ValueError:
        raise SerpApiError("SerpApi returned an unreadable response.") from None
    if not isinstance(payload, dict):
        raise SerpApiError("SerpApi returned an unexpected response format.")
    metadata = payload.get("search_metadata")
    if payload.get("error") or (isinstance(metadata, dict) and metadata.get("status") == "Error"):
        raise SerpApiError("SerpApi could not complete the Google Shopping search. Check the query and account quota.")
    return payload


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _http_url(value: object) -> str:
    url = _text(value)
    if not url or re.search(r"[\s\x00-\x1f\x7f\\]", url):
        return ""
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            return ""
        parsed.port  # Reject malformed ports and IPv6 authorities.
    except ValueError:
        return ""
    return url


def _retailer_url(value: object) -> str:
    """Accept only apparent direct links; never follow third-party redirects."""
    url = _http_url(value)
    if not url:
        return ""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    labels = host.split(".")
    if any(label in {"google", "googleadservices", "googlesyndication", "doubleclick", "serpapi"} for label in labels):
        return ""
    blocked = ("g.co", "goo.gl", "bit.ly", "t.co", "tinyurl.com", "awin1.com", "anrdoezrs.net", "jdoqocy.com", "tkqlhce.com", "linksynergy.com", "shareasale.com")
    if any(host == item or host.endswith("." + item) for item in blocked):
        return ""
    if re.search(r"(?:^|/)(?:url|redirect|redirecting|out|click|track)(?:/|$)", parsed.path, re.I):
        return ""
    redirect_keys = {"url", "uri", "redirect", "redirect_url", "redirect_uri", "redirect_to", "destination", "dest", "target", "u"}
    if any(key.lower() in redirect_keys for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        return ""
    return url


def _shopping_url(value: object) -> str:
    """Return a Google Shopping result page, never a merchant link or redirect."""
    url = _http_url(value)
    if not url:
        return ""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme.lower() != "https" or host not in {"google.com", "www.google.com", "shopping.google.com"}:
        return ""
    if not (parsed.path == "/search" or parsed.path.startswith("/shopping/")):
        return ""
    return url


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _price_and_currency(hit: dict) -> tuple[float | None, str | None, str]:
    display = _text(hit.get("price"))
    # A range, installment, or qualified price is not a single offer price.
    simple = re.fullmatch(r"\s*(?:(?:[A-Za-z]{3}|US\$|CA\$|C\$|AU\$|A\$|[$€£₹¥])\s*)?([0-9][0-9.,\s\u00a0\u202f]*)(?:\s*(?:[A-Za-z]{3}|[$€£₹¥]))?\s*", display, re.I)
    if display and not simple:
        return None, None, display
    currency = None
    explicit = _text(hit.get("currency")).upper()
    codes = {"USD", "CAD", "AUD", "NZD", "EUR", "GBP", "INR", "JPY", "CNY", "CHF", "PLN", "SEK", "NOK", "DKK", "BRL", "MXN"}
    tokens = set(re.findall(r"(?<![A-Za-z])(?:USD|CAD|AUD|NZD|EUR|GBP|INR|JPY|CNY|CHF|PLN|SEK|NOK|DKK|BRL|MXN)(?![A-Za-z])", display, re.I))
    identified = {token.upper() for token in tokens}
    prefix = re.match(r"\s*(US\$|CA\$|C\$|AU\$|A\$)", display, re.I)
    if prefix:
        identified.add({"US$": "USD", "CA$": "CAD", "C$": "CAD", "AU$": "AUD", "A$": "AUD"}[prefix.group(1).upper()])
    for symbol, code in (("€", "EUR"), ("£", "GBP"), ("₹", "INR")):
        if symbol in display:
            identified.add(code)
    if explicit in codes:
        identified.add(explicit)
    if len(identified) == 1:
        currency = identified.pop()
    price = _number(hit.get("extracted_price"))
    if price is None and simple:
        raw_digits = simple.group(1).strip()
        if re.search(r"\s", raw_digits) and not re.fullmatch(r"\d{1,3}(?:[\s\u00a0\u202f]\d{3})+(?:[.,]\d{1,2})?", raw_digits):
            return None, currency, display
        digits = re.sub(r"[\s\u00a0\u202f]", "", raw_digits)
        if re.fullmatch(r"\d+", digits):
            price = _number(digits)
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d{1,2}|\d+\.\d{1,2}", digits):
            price = _number(digits.replace(",", ""))
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{1,2}|\d+,\d{1,2}", digits):
            price = _number(digits.replace(".", "").replace(",", "."))
        # A single separator followed by three digits is locale-ambiguous.
    return price, currency, display


def _review_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"\d+|\d{1,3}(?:,\d{3})+", value.strip()):
        return int(value.strip().replace(",", ""))
    return None


def _normalize(hit: dict) -> dict | None:
    name = _text(hit.get("title"))
    if not name:
        return None
    price, currency, display = _price_and_currency(hit)
    rating = _number(hit.get("rating"))
    if rating is not None and rating > 5:
        rating = None
    features = []
    for field in ("features", "extensions"):
        values = hit.get(field)
        if isinstance(values, list):
            for item in values:
                text = _text(item)
                if text and text not in features:
                    features.append(text)
    specs = hit.get("specifications")
    specifications = {}
    if isinstance(specs, dict):
        for key, value in specs.items():
            if isinstance(key, str) and key.strip() and isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
                specifications[key.strip()] = str(value).strip()
    note = "Google Shopping search result via SerpApi; verify price, seller, and availability. Ratings may describe the product rather than the seller."
    if hit.get("multiple_sources") is True:
        note += " This result may include multiple sellers."
    return {
        "name": name,
        "url": next((url for key in ("link", "product_link") if (url := _retailer_url(hit.get(key)))), ""),
        "shopping_url": next((url for key in ("product_link", "link") if (url := _shopping_url(hit.get(key)))), ""),
        "image_url": _http_url(hit.get("thumbnail")),
        "price": price,
        "price_display": display,
        "currency": currency,
        "rating": rating,
        "review_count": _review_count(hit.get("reviews")),
        "seller": _text(hit.get("source")),
        "description": _text(hit.get("description")) or _text(hit.get("snippet")),
        "features": features,
        "specifications": specifications,
        "source_note": note,
        "category": _text(hit.get("category")),
    }


def search_shopping_products(query: str, limit: int = 40) -> list[dict]:
    """Return normalized Shopping results, up to limit, in result order.

    IDs are content-derived result identifiers, not merchant product IDs. Exact
    duplicates are removed only when a retailer URL or seller is available;
    indistinguishable anonymous results are retained with occurrence suffixes.
    """
    if not isinstance(query, str) or not (clean_query := " ".join(query.split())):
        raise ValueError("Enter a product search phrase.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("The result limit must be a positive integer.")
    payload = _serpapi_request({"engine": "google_shopping", "q": clean_query})
    hits = payload.get("shopping_results", [])
    if not isinstance(hits, list):
        raise SerpApiError("SerpApi returned an unexpected shopping results format.")
    results = []
    seen: dict[str, int] = {}
    for hit in hits:
        product = _normalize(hit) if isinstance(hit, dict) else None
        if product is None:
            continue
        identity = json.dumps(product, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        occurrence = seen.get(digest, 0)
        if occurrence and (product["url"] or product["seller"]):
            continue
        seen[digest] = occurrence + 1
        product["id"] = f"shopping-{digest}" + (f"-{occurrence + 1}" if occurrence else "")
        results.append(product)
        if len(results) >= limit:
            break
    return results
