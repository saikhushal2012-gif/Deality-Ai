"""Retailer-independent, deterministic comparisons of two product mappings.

``comparison_compatibility`` returns a status, evidence and explanation; it
does not independently verify listings. Both comparison functions reject
incompatible types. For an uncertain result, a caller must obtain the user's
confirmation outside this module and pass ``confirmed=True``. Confirmation
cannot override an incompatible result. No input is changed or fetched.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


PRIORITIES = {
    "Balanced value": {"price": 0.5, "rating": 0.5},
    "Lower price": {"price": 1.0},
    "Higher customer rating": {"rating": 1.0},
    "More customer reviews": {"reviews": 1.0},
    "Price-focused value": {"price": 0.7, "rating": 0.3},
    "Rating-focused value": {"price": 0.3, "rating": 0.7},
}

SPEC_FOCUSES = {
    "Materials & build": ("material", "construction", "case", "frame", "finish", "fabric", "build"),
    "Battery & power": ("battery", "power", "charging", "charge", "voltage", "watt"),
    "Performance & components": ("processor", "cpu", "gpu", "memory", "ram", "storage", "motor", "chip", "component", "performance", "speed"),
    "Size & weight": ("size", "weight", "dimension", "length", "width", "height"),
    "Warranty & support": ("warranty", "support", "return", "guarantee"),
}

# Each key represents a specific purpose, not just a broad retail department.
# Computers are deliberately grouped; musical instruments retain subtypes.
_ALIASES = {
    "watch": ("watch", "watches", "wristwatch", "wristwatches", "smartwatch", "smartwatches", "smart watch", "smart watches"),
    "computer": ("computer", "computers", "laptop", "laptops", "laptop computer", "notebook computer", "desktop", "desktops", "desktop computer", "desktop pc", "gaming pc", "pc", "pcs", "macbook"),
    "smartphone": ("smartphone", "smartphones", "phone", "phones", "mobile phone", "mobile phones", "cell phone", "cell phones", "cellphone", "cellphones", "iphone", "iphones"),
    "hoodie": ("hoodie", "hoodies", "hooded sweatshirt", "hooded sweatshirts"),
    "guitar": ("guitar", "guitars"),
    "ukulele": ("ukulele", "ukuleles"),
    "piano": ("piano", "pianos", "digital piano", "digital pianos"),
    "violin": ("violin", "violins"),
    "drum kit": ("drum kit", "drum kits", "drum set", "drum sets"),
    "refrigerator": ("refrigerator", "refrigerators", "fridge", "fridges"),
    "washing machine": ("washing machine", "washing machines", "clothes washer", "clothes washers"),
    "clothes dryer": ("clothes dryer", "clothes dryers", "tumble dryer", "tumble dryers"),
    "microwave": ("microwave", "microwaves", "microwave oven", "microwave ovens"),
    "blender": ("blender", "blenders"),
    "toaster": ("toaster", "toasters"),
    "vacuum cleaner": ("vacuum cleaner", "vacuum cleaners", "robot vacuum", "robot vacuums"),
    "television": ("television", "televisions", "tv", "tvs", "smart tv", "smart tvs"),
    "headphones": ("headphone", "headphones", "earbud", "earbuds", "earphone", "earphones"),
    "camera": ("camera", "cameras"),
    "tablet": ("tablet", "tablets", "ipad", "ipads"),
}
_INSTRUMENTS = {"guitar", "ukulele", "piano", "violin", "drum kit"}
_CATEGORY_ALIASES = {alias: kind for kind, aliases in _ALIASES.items() for alias in aliases}
_NAME_PATTERNS = {
    kind: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in aliases) + r")(?!\w)")
    for kind, aliases in _ALIASES.items()
}
# Mentioning a product in an accessory title is not evidence that it is the product.
_ACCESSORY = re.compile(r"\b(?:accessor(?:y|ies)|case|cases|cover|covers|strap|straps|band|bands|screen protector|charger|chargers|replacement|spare|bag|bags|sleeve|sleeves|stand|stands|mount|mounts|strings|cable|cables|parts)\b")


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"[_\-]+", " ", value.casefold()).split())


def _type_evidence(product: Mapping[str, Any]) -> dict[str, Any]:
    category = _normalize(product.get("category"))
    name = _normalize(product.get("name"))
    category_type = _CATEGORY_ALIASES.get(category)
    matches = {kind for kind, pattern in _NAME_PATTERNS.items() if pattern.search(name)}
    # Accessory and multi-product titles are ambiguous even with a category.
    if _ACCESSORY.search(category) or _ACCESSORY.search(name):
        return {"type": None, "source": None, "reason": "Possible accessory or replacement part"}
    if len(matches) > 1 or (category_type and matches and category_type not in matches):
        return {"type": None, "source": None, "reason": "Conflicting product-type clues"}
    if category_type:
        return {"type": category_type, "source": "category", "reason": "Recognized category alias"}
    if matches:
        return {"type": next(iter(matches)), "source": "name", "reason": "Recognized name alias"}
    return {"type": None, "source": None, "reason": "No unambiguous recognized type"}


def _pair_compatibility(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``status``, ``types``, ``evidence`` and a human-readable ``reason``.

    Status is ``compatible`` for the same recognized purpose, ``incompatible``
    for clearly different recognized purposes, otherwise ``uncertain``. Unlike
    instruments are uncertain rather than automatically interchangeable. Types
    are inferred alias labels or None, not independently verified facts. Only
    category and name are used; description and features may describe other
    products. This deliberately does not infer a type for unknown text.
    """
    evidence = [_type_evidence(first), _type_evidence(second)]
    types = [item["type"] for item in evidence]
    if None in types:
        status = "uncertain"
        reason = "At least one product type is unknown, ambiguous, or conflicting; user confirmation is required."
    elif types[0] == types[1]:
        status = "compatible"
        reason = "Both listings match the same recognized product purpose; listing details are not independently verified."
    elif all(kind in _INSTRUMENTS for kind in types):
        status = "uncertain"
        reason = "Different musical instrument types may have different purposes; user confirmation is required."
    else:
        status = "incompatible"
        reason = "The listings match different recognized product purposes and cannot be compared."
    return {"status": status, "types": types, "evidence": evidence, "reason": reason}


def comparison_compatibility(first: Mapping[str, Any], second: Mapping[str, Any], *others: Mapping[str, Any]) -> dict[str, Any]:
    """Check every product pair and summarize compatibility for 2–5 listings."""
    products = (first, second, *others)
    evidence = [_type_evidence(product) for product in products]
    types = [item["type"] for item in evidence]
    if len(products) > 5:
        return {"status": "incompatible", "types": types, "evidence": evidence,
                "reason": "Compare between two and five products at a time."}
    pair_results = [_pair_compatibility(products[i], products[j])
                    for i in range(len(products)) for j in range(i + 1, len(products))]
    if any(result["status"] == "incompatible" for result in pair_results):
        status = "incompatible"
        reason = "The selected listings include different recognized product purposes and cannot be compared together."
    elif any(result["status"] == "uncertain" for result in pair_results):
        status = "uncertain"
        reason = "At least one product type or relationship is uncertain; user confirmation is required that all selected products serve the same purpose."
    else:
        status = "compatible"
        reason = "All selected listings match the same recognized product purpose; listing details are not independently verified."
    return {"status": status, "types": types, "evidence": evidence, "reason": reason}


def _require_products(products: Sequence[Mapping[str, Any]], confirmed: bool) -> None:
    result = comparison_compatibility(products[0], products[1], *products[2:])
    if result["status"] == "incompatible" or (result["status"] == "uncertain" and confirmed is not True):
        raise ValueError(result["reason"])


def _require_compatibility(first: Mapping[str, Any], second: Mapping[str, Any], confirmed: bool) -> None:
    _require_products((first, second), confirmed)


def _number(value: Any, *, maximum: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value.replace(",", "") if isinstance(value, str) else value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        return None
    return number


def _text(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "Not listed"
    return str(value)


def _currency(value: Any) -> str:
    code = value.strip().upper() if isinstance(value, str) else ""
    return code if re.fullmatch(r"[A-Z]{3}", code) else ""


def comparison_rows(products: Sequence[Mapping[str, Any]], *, confirmed: bool = False) -> list[dict[str, str]]:
    """Return side-by-side rows for two to five compatible product mappings.

    ``confirmed=True`` permits an uncertain pair after external confirmation.
    Spec keys match case-insensitively after underscores and whitespace are
    normalized. Distinct colliding values are retained; numbered feature specs
    become features rather than being compared by position. Missing data is
    shown as ``Not listed``. No currency conversion or spec ranking is done.
    """
    if not 2 <= len(products) <= 5:
        raise ValueError("Between two and five products are required.")
    _require_products(products, confirmed)
    option_columns = [f"Option {index + 1}" for index in range(len(products))]

    def row(label: str, values: Sequence[str]) -> dict[str, str]:
        return {"Detail": label, **dict(zip(option_columns, values))}

    rows = []
    for label, key in (("Name", "name"), ("URL", "url"), ("Image URL", "image_url"),
                       ("Category", "category"), ("Price", "price_display"),
                       ("Rating", "rating"), ("Reviews", "review_count"),
                       ("Seller", "seller"), ("Description", "description")):
        values = []
        for product in products:
            value = product.get(key)
            if key == "price_display" and not value:
                price, currency = _number(product.get("price")), _currency(product.get("currency"))
                value = f"{currency} {price:g}" if price is not None and currency else None
            elif key == "rating":
                number = _number(value, maximum=5)
                value = f"{number:g}/5" if number is not None else None
            elif key == "review_count":
                number = _number(value)
                value = f"{number:.0f}" if number is not None and number.is_integer() else None
            values.append(_text(value))
        rows.append(row(label, values))

    specs: dict[str, dict[str, Any]] = {}
    features: list[list[str]] = [[] for _ in products]

    def add_feature(index: int, value: Any) -> None:
        text = _text(value)
        if text != "Not listed" and text not in features[index]:
            features[index].append(text)

    for index, product in enumerate(products):
        items = product.get("features")
        if isinstance(items, str):
            items = [items]
        if isinstance(items, Sequence):
            for item in items:
                add_feature(index, item)
        specifications = product.get("specifications")
        if not isinstance(specifications, Mapping):
            continue
        for key, value in specifications.items():
            normalized = " ".join(str(key).replace("_", " ").split()).casefold()
            if re.fullmatch(r"feature\s+\d+", normalized):
                add_feature(index, value)
                continue
            entry = specs.setdefault(normalized, {"label": str(key), "values": [[] for _ in products]})
            text = _text(value)
            if text != "Not listed" and text not in entry["values"][index]:
                entry["values"][index].append(text)
    for entry in specs.values():
        rows.append(row(entry["label"], ["\n".join(values) if values else "Not listed" for values in entry["values"]]))
    rows.append(row("Features", ["\n".join(values) if values else "Not listed" for values in features]))
    return rows


def recommend_products(first: Mapping[str, Any], second: Mapping[str, Any], priority: str, *, confirmed: bool = False) -> dict[str, Any]:
    """Return ``scores``, zero-based ``winner_index``, ``score_text``, and explanation.

    Numeric priorities use only dimensions present and valid for both items,
    renormalizing remaining weights. Prices need matching explicit three-letter
    currency codes; ratings are out of five; review counts must be integers.
    Scores are relative to this pair and ties have no winner. Spec focuses show
    matching listing evidence without inventing a winner. Unknown priorities
    raise ValueError. Compatibility and confirmation rules match comparison_rows.
    """
    if priority not in PRIORITIES and priority not in SPEC_FOCUSES:
        raise ValueError(f"Unknown recommendation priority: {priority!r}")
    _require_compatibility(first, second, confirmed)
    if priority in SPEC_FOCUSES:
        rows = comparison_rows((first, second), confirmed=confirmed)
        matches = [row for row in rows if any(term in " ".join(row.values()).casefold() for term in SPEC_FOCUSES[priority])]
        evidence = "; ".join(f"{row['Detail']}: {row['Option 1']} vs {row['Option 2']}" for row in matches)
        return {"scores": [None, None], "winner_index": None, "score_text": "Option 1: not scored · Option 2: not scored",
                "explanation": "No winner: specifications and features are not scored. " + (evidence or "No matching listing evidence was provided.")}

    products = (first, second)
    desired = PRIORITIES[priority]
    dimensions: dict[str, list[float]] = {}
    omitted: list[str] = []
    if "price" in desired:
        prices = [_number(product.get("price")) for product in products]
        currencies = [_currency(product.get("currency")) for product in products]
        if any(value is None for value in prices):
            omitted.append("price (missing or invalid price)")
        elif not all(currencies):
            omitted.append("price (missing or invalid currency code)")
        elif currencies[0] != currencies[1]:
            omitted.append("price (different currencies; no conversion performed)")
        else:
            low = min(prices)
            dimensions["price"] = [100.0 * (low / value) if value else 100.0 for value in prices]
    if "rating" in desired:
        ratings = [_number(product.get("rating"), maximum=5) for product in products]
        if any(value is None for value in ratings):
            omitted.append("rating (missing or invalid rating)")
        else:
            dimensions["rating"] = [value * 20.0 for value in ratings]
    if "reviews" in desired:
        reviews = [_number(product.get("review_count")) for product in products]
        if any(value is None or not value.is_integer() for value in reviews):
            omitted.append("reviews (missing or invalid review count)")
        else:
            high = max(reviews)
            dimensions["reviews"] = [100.0 * (value / high) if high else 100.0 for value in reviews]

    total_weight = sum(desired[key] for key in dimensions)
    scores = [sum(dimensions[key][index] * desired[key] / total_weight for key in dimensions)
              for index in range(2)] if total_weight else [None, None]
    if not total_weight:
        winner, outcome = None, "No winner: insufficient comparable data for this priority."
    elif math.isclose(scores[0], scores[1], rel_tol=0.0, abs_tol=1e-9):
        winner, outcome = None, "No winner: the comparable scores are tied."
    else:
        winner = 0 if scores[0] > scores[1] else 1
        outcome = f"Option {winner + 1} has the higher comparable score."
    score_text = " · ".join(f"Option {i + 1}: {score:.2f}/100" if score is not None
                            else f"Option {i + 1}: insufficient data" for i, score in enumerate(scores))
    formulas = {"price": "100 × lowest price / price (zero price: 100, positive price against zero: 0)",
                "rating": "rating / 5 × 100", "reviews": "100 × review count / largest count (both zero: 100 each)"}
    details = [f"{key}: {formulas[key]}; option scores {values[0]:.2f}, {values[1]:.2f}; effective weight {desired[key] / total_weight:.0%}"
               for key, values in dimensions.items()]
    explanation = f"{outcome} {score_text}. "
    if details:
        explanation += "Scoring: " + "; ".join(details) + ". "
    if omitted:
        explanation += "Excluded for both options: " + "; ".join(omitted) + ". "
    if "reviews" in desired:
        explanation += "Review count is not a measure of quality. "
    explanation += "Scores are comparison-local; specifications and features are not scored."
    return {"scores": scores, "winner_index": winner, "score_text": score_text, "explanation": explanation}


def recommend_many(products: Sequence[Mapping[str, Any]], priority: str, *, confirmed: bool = False) -> dict[str, Any]:
    """Compare 2–5 listings using the same evidence-based rules as pair scoring."""
    if not 2 <= len(products) <= 5:
        raise ValueError("Between two and five products are required.")
    if len(products) == 2:
        return recommend_products(products[0], products[1], priority, confirmed=confirmed)
    if priority not in PRIORITIES and priority not in SPEC_FOCUSES:
        raise ValueError(f"Unknown recommendation priority: {priority!r}")
    _require_products(products, confirmed)
    option_columns = [f"Option {index + 1}" for index in range(len(products))]
    if priority in SPEC_FOCUSES:
        rows = comparison_rows(products, confirmed=confirmed)
        matches = [row for row in rows if any(term in " ".join(row.values()).casefold()
                                               for term in SPEC_FOCUSES[priority])]
        evidence = "; ".join(f"{row['Detail']}: " + " vs ".join(row[key] for key in option_columns)
                              for row in matches)
        score_text = " · ".join(f"{column}: not scored" for column in option_columns)
        return {"scores": [None] * len(products), "winner_index": None, "score_text": score_text,
                "explanation": "No winner: specifications and features are not scored. " +
                (evidence or "No matching listing evidence was provided.")}

    desired = PRIORITIES[priority]
    dimensions: dict[str, list[float]] = {}
    omitted: list[str] = []
    if "price" in desired:
        prices = [_number(product.get("price")) for product in products]
        currencies = [_currency(product.get("currency")) for product in products]
        if any(value is None for value in prices):
            omitted.append("price (missing or invalid price)")
        elif not all(currencies):
            omitted.append("price (missing or invalid currency code)")
        elif len(set(currencies)) != 1:
            omitted.append("price (different currencies; no conversion performed)")
        else:
            low = min(prices)
            dimensions["price"] = [100.0 * (low / value) if value else 100.0 for value in prices]
    if "rating" in desired:
        ratings = [_number(product.get("rating"), maximum=5) for product in products]
        if any(value is None for value in ratings):
            omitted.append("rating (missing or invalid rating)")
        else:
            dimensions["rating"] = [value * 20.0 for value in ratings]
    if "reviews" in desired:
        reviews = [_number(product.get("review_count")) for product in products]
        if any(value is None or not value.is_integer() for value in reviews):
            omitted.append("reviews (missing or invalid review count)")
        else:
            high = max(reviews)
            dimensions["reviews"] = [100.0 * (value / high) if high else 100.0 for value in reviews]

    total_weight = sum(desired[key] for key in dimensions)
    scores = [sum(dimensions[key][index] * desired[key] / total_weight for key in dimensions)
              for index in range(len(products))] if total_weight else [None] * len(products)
    if not total_weight:
        winner, outcome = None, "No winner: insufficient comparable data for this priority."
    else:
        high_score = max(scores)
        leaders = [index for index, score in enumerate(scores)
                   if math.isclose(score, high_score, rel_tol=0.0, abs_tol=1e-9)]
        if len(leaders) != 1:
            winner, outcome = None, "No winner: the highest comparable scores are tied."
        else:
            winner = leaders[0]
            outcome = f"Option {winner + 1} has the highest comparable score."
    score_text = " · ".join(f"{column}: {score:.2f}/100" if score is not None
                            else f"{column}: insufficient data"
                            for column, score in zip(option_columns, scores))
    explanation = f"{outcome} {score_text}. Scores are comparison-local."
    if omitted:
        explanation += " Excluded dimensions: " + "; ".join(omitted) + "."
    if "reviews" in desired:
        explanation += " Review count is not a measure of quality."
    explanation += " Specifications and features are not scored."
    return {"scores": scores, "winner_index": winner, "score_text": score_text, "explanation": explanation}