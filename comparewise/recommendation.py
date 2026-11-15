"""Transparent, comparison-local ranking from the product details users provide."""

from __future__ import annotations

from typing import Iterable

from .schemas import Product, infer_product_type, normalize_product_type

FEATURES = ("price", "performance", "quality", "rating", "features", "bundle", "value")
FEATURE_LABELS = {
    "price": "lower price", "performance": "performance", "quality": "quality",
    "rating": "customer rating", "features": "useful features",
    "bundle": "setup fit", "value": "overall value",
}
DEFAULT_PRIORITIES = {
    "price": 0.25, "performance": 0.15, "quality": 0.20,
    "rating": 0.10, "features": 0.15, "bundle": 0.0, "value": 0.15,
}


def _normalize_priorities(priorities: dict[str, float] | None) -> dict[str, float]:
    source = priorities or DEFAULT_PRIORITIES
    weights = {key: max(0.0, float(source.get(key, 0.0))) for key in FEATURES}
    total = sum(weights.values())
    if total <= 0:
        return {key: value for key, value in DEFAULT_PRIORITIES.items()}
    return {key: value / total for key, value in weights.items()}


def _matched_bundle_items(product: Product, needs: Iterable[str], owned: Iterable[str]) -> set[str]:
    included = {item.strip().casefold() for item in product.included_items if item.strip()}
    needed = {item.strip().casefold() for item in needs if item.strip()}
    already_owned = {item.strip().casefold() for item in owned if item.strip()}
    return (included & needed) - already_owned


def recommend(
    products: list[Product],
    priorities: dict[str, float] | None = None,
    setup_needs: Iterable[str] = (),
    accessories_owned: Iterable[str] = (),
) -> dict:
    """Score current products directly; no fabricated labels or background model fit."""
    if not products:
        return {"ranked": [], "best_overall": None, "best_budget": None, "best_performance": None}

    product_types = {normalize_product_type(product.category) for product in products}
    if "" in product_types or len(product_types) > 1:
        raise ValueError("Compare only products of the same type or purpose. Start a new comparison to change product type.")
    for product in products:
        inferred_type = infer_product_type(f"{product.name} {product.description}")
        if inferred_type and inferred_type != normalize_product_type(product.category):
            raise ValueError(f"{product.name} appears to be {inferred_type}, not {product.category}. Remove it or start a comparison for the matching product type.")

    currencies = {p.currency.upper() for p in products if p.price is not None and p.currency}
    if len(currencies) > 1:
        raise ValueError("Products have different currencies. Convert prices to one currency before ranking.")

    priced = [p.price for p in products if p.price is not None and p.price >= 0]
    low, high = (min(priced), max(priced)) if priced else (None, None)
    needs, owned = list(setup_needs), list(accessories_owned)
    remaining_needs = {x.strip().casefold() for x in needs if x.strip()} - {x.strip().casefold() for x in owned if x.strip()}
    weights = _normalize_priorities(priorities)

    ranked = []
    for product in products:
        features: dict[str, float] = {}
        missing = []
        if product.price is None:
            missing.append("price")
        elif low is not None and high is not None:
            features["price"] = 1.0 if high == low else 1.0 - (float(product.price) - low) / (high - low)

        for key, value, label, maximum in (
            ("performance", product.performance, "performance", 100.0),
            ("quality", product.quality, "quality", 100.0),
            ("rating", product.rating, "rating", 5.0),
            ("features", product.features_score, "feature usefulness", 100.0),
        ):
            if value is None:
                missing.append(label)
            else:
                features[key] = max(0.0, min(1.0, float(value) / maximum))

        matched = _matched_bundle_items(product, needs, owned)
        if remaining_needs:
            features["bundle"] = len(matched) / len(remaining_needs)

        # Value is derived only if price, performance, and quality are all known.
        if "price" in features and "performance" in features and "quality" in features:
            features["value"] = features["price"] * (0.55 * features["performance"] + 0.45 * features["quality"])
        elif product.performance is None or product.quality is None:
            missing.append("overall value (needs price, performance, and quality)")

        active_weight = sum(weights[key] for key in features)
        if active_weight > 0:
            score = 100.0 * sum(features[key] * weights[key] for key in features) / active_weight
        else:
            score = None

        contributions = sorted(
            ((key, features[key] * weights[key] / active_weight) for key in features),
            key=lambda item: item[1], reverse=True,
        ) if active_weight else []
        reasons = [f"{FEATURE_LABELS[key]} matches your priority" for key, amount in contributions[:2] if amount > 0]
        if matched:
            reasons.append("includes needed items: " + ", ".join(sorted(matched)))
        if missing:
            reasons.append("unknown: " + ", ".join(missing))
        if score is None:
            reasons = ["not enough comparable information to recommend this product"]

        ranked.append({
            "name": product.name,
            "score": round(score, 1) if score is not None else None,
            "price": product.price,
            "currency": product.currency,
            "features": [features.get(key) for key in FEATURES],
            "missing_fields": missing,
            "matched_items": sorted(matched),
            "reasons": reasons,
            "product": product,
        })

    ranked.sort(key=lambda result: (result["score"] is not None, result["score"] or 0.0), reverse=True)
    scored = [result for result in ranked if result["score"] is not None]
    budget_products = [p for p in products if p.price is not None]
    measured_performance = [p for p in products if p.performance is not None]
    return {
        "ranked": ranked,
        "best_overall": scored[0]["name"] if scored else None,
        "best_budget": min(budget_products, key=lambda p: p.price).name if budget_products else None,
        "best_performance": max(measured_performance, key=lambda p: p.performance).name if measured_performance else None,
        "priorities": weights,
        "model": "direct weighted scoring of this comparison; no model training is performed",
        "metrics": None,
    }


def evaluation_metrics() -> dict:
    """Synthetic metrics were removed; real preference labels are not available yet."""
    return {
        "status": "not evaluated",
        "reason": "Recommendations use direct comparison inputs. Real preference outcomes are needed before fitting and evaluating a learned recommendation model.",
    }
