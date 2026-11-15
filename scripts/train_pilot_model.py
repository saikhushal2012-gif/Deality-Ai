"""Train and evaluate the experimental Deality Match preference model.

Only scenarios with exactly one explicit ``user_preferred=yes`` and at least
one explicit ``no`` are eligible. Tentative, provisional, and no-winner labels
are excluded. Evaluation holds out whole scenarios and compares top-choice
accuracy with the cheapest-candidate baseline. A tiny scenario count is not a
valid performance claim. Deality Match is not a production recommender.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "labeling_pilot.csv"
DEFAULT_ADDITIONAL_DATA = ROOT / "data" / "deality_match_scenarios.json"
DEFAULT_OTHER_DATA = ROOT / "data" / "deality_match_other_scenarios.json"
DEFAULT_MODEL = ROOT / "artifacts" / "laptop_preference_pilot.joblib"
TEXT_FIELDS = (
    "shopper_profile", "hard_requirements", "priorities", "product_name",
    "ram", "cpu", "gpu", "storage", "display", "battery_life", "weight",
    "product_category", "product_attributes",
)


def load_labeled_scenarios(path: Path) -> dict[str, list[dict[str, str]]]:
    """Load only explicit preference labels with one winner per scenario."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("user_preferred") in {"yes", "no"}:
                row.setdefault("product_category", "laptop")
                grouped[row["scenario_id"]].append(row)

    eligible = {}
    for scenario_id, rows in grouped.items():
        labels = [row["user_preferred"] for row in rows]
        if labels.count("yes") == 1 and labels.count("no") >= 1:
            eligible[scenario_id] = rows
    return eligible


def load_additional_scenarios(path: Path = DEFAULT_ADDITIONAL_DATA) -> dict[str, list[dict[str, str]]]:
    """Convert explicitly chosen, user-provided scenarios to pilot rows."""
    import json

    scenarios: dict[str, list[dict[str, str]]] = {}
    for scenario in json.loads(path.read_text(encoding="utf-8")):
        scenario_id = str(scenario["scenario_id"])
        candidates = scenario.get("candidates", [])
        preferred = scenario.get("preferred")
        if not scenario_id or not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"{scenario_id or 'An additional scenario'} must have at least two candidates.")
        if sum(candidate.get("name") == preferred for candidate in candidates) != 1:
            raise ValueError(f"{scenario_id} must have exactly one explicit preferred candidate.")

        rows = []
        category = str(scenario.get("category", "laptop"))
        for candidate in candidates:
            attributes = candidate.get("attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}
            rows.append({
                "scenario_id": scenario_id,
                "shopper_profile": str(scenario.get("shopper_profile", "")),
                "budget_max": str(scenario.get("budget", "")),
                "budget_currency": str(scenario.get("currency", "")),
                "hard_requirements": str(scenario.get("hard_requirements", "")),
                "priorities": "; ".join(str(value) for value in scenario.get("priorities", [])),
                "product_name": str(candidate.get("name", "")),
                "price": str(candidate.get("price", "")),
                "price_currency": str(scenario.get("currency", "")),
                "ram": str(candidate.get("ram", "")),
                "cpu": str(candidate.get("cpu", "")),
                "gpu": str(candidate.get("gpu", "")),
                "storage": str(candidate.get("storage", "")),
                "display": str(candidate.get("display", "")),
                "battery_life": str(candidate.get("battery_life", "")),
                "weight": str(candidate.get("weight", "")),
                "source_retailer": str(attributes.get("retailer", "Not stated")),
                "source_note": "User-provided scenario and listing details; not independently verified.",
                "meets_hard_needs": "",
                "user_preferred": "yes" if candidate.get("name") == preferred else "no",
                "preference_reason": str(scenario.get("reason", "")),
                "review_status": "user_supplied_unverified",
                "product_category": category,
                "product_attributes": "; ".join(
                    f"{key}: {value}" for key, value in sorted(attributes.items())
                ),
            })
        scenarios[scenario_id] = rows
    return scenarios


def load_deality_match_scenarios(
    pilot_path: Path = DEFAULT_DATA,
    additional_path: Path = DEFAULT_ADDITIONAL_DATA,
    other_path: Path = DEFAULT_OTHER_DATA,
) -> dict[str, list[dict[str, str]]]:
    """Combine original pilot, later laptop, and other-category scenarios."""
    scenarios = load_labeled_scenarios(pilot_path)
    for path in (additional_path, other_path):
        for scenario_id, rows in load_additional_scenarios(path).items():
            if scenario_id in scenarios:
                raise ValueError(f"Scenario ID is duplicated across datasets: {scenario_id}")
            scenarios[scenario_id] = rows
    return scenarios


def row_text(row: dict[str, str]) -> str:
    """Build text features without including target labels or human rationale."""
    parts = [f"{field}: {row.get(field, '')}" for field in TEXT_FIELDS]
    try:
        price = float(row.get("price", ""))
        budget = float(row.get("budget_max", ""))
    except (TypeError, ValueError):
        price = budget = 0.0
    if price > 0:
        parts.append(f"price {price:.0f} currency {row.get('price_currency', '')}")
    if price > 0 and budget > 0:
        parts.append("within budget" if price <= budget else "over budget")
    return " ".join(parts)


def make_model() -> Pipeline:
    return Pipeline([
        ("text", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)),
    ])


_LAPTOP_PATTERN = re.compile(r"\b(?:laptop|notebook)\b", re.I)
_MODEL_CATEGORY_PATTERNS = {
    "laptop": re.compile(r"\b(?:laptop|notebook)\b", re.I),
    "smartphone": re.compile(r"\b(?:smartphone|cell phone|mobile phone|iphone|pixel|galaxy [as]\d)\b", re.I),
    "smartwatch": re.compile(r"\b(?:smartwatch|smart watch|fitness watch|garmin forerunner|apple watch)\b", re.I),
    "hoodie": re.compile(r"\bhoodie\b", re.I),
    "shorts": re.compile(r"\bshorts?\b", re.I),
    "headphones": re.compile(r"\b(?:headphones?|earbuds?|headset)\b", re.I),
    "sneakers": re.compile(r"\b(?:sneakers?|running shoes?|shoes|footwear)\b", re.I),
    "portable_speaker": re.compile(r"\b(?:portable )?(?:bluetooth )?speakers?\b", re.I),
    "ereader": re.compile(r"\b(?:e[ -]?reader|kindle|kobo)\b", re.I),
    "power_bank": re.compile(r"\b(?:power ?bank|portable charger|battery pack)\b", re.I),
    "backpack": re.compile(r"\bbackpacks?\b", re.I),
    "rain_jacket": re.compile(r"\b(?:rain jacket|raincoat|shell jacket|rain shell)\b", re.I),
    "air_fryer": re.compile(r"\bair fryers?\b", re.I),
}
_SPEC_FIELDS = {
    "ram": ("ram", "memory"),
    "cpu": ("cpu", "processor"),
    "gpu": ("gpu", "graphics"),
    "storage": ("storage", "ssd", "hard drive"),
    "display": ("display", "screen"),
    "battery_life": ("battery life", "battery"),
    "weight": ("weight",),
}


def infer_model_category(product: dict[str, Any]) -> str | None:
    text = str(product.get("category", "")) + " " + str(product.get("name", ""))
    matches = [category for category, pattern in _MODEL_CATEGORY_PATTERNS.items() if pattern.search(text)]
    return matches[0] if len(matches) == 1 else None


def supports_deality_match_products(products: list[dict[str, Any]]) -> bool:
    if not 2 <= len(products) <= 5:
        return False
    categories = [infer_model_category(product) for product in products]
    if categories[0] is None or not all(category == categories[0] for category in categories):
        return False
    available_categories = {
        row.get("product_category")
        for rows in load_deality_match_scenarios().values()
        for row in rows
    }
    return categories[0] in available_categories


def supports_laptop_products(products: list[dict[str, Any]]) -> bool:
    return 2 <= len(products) <= 5 and all(infer_model_category(product) == "laptop" for product in products)


def _candidate_row(product: dict[str, Any], priority: str, chat: list[dict[str, Any]]) -> dict[str, str]:
    user_context = " ".join(
        str(message.get("content", ""))
        for message in chat
        if isinstance(message, dict) and message.get("role") == "user"
    )
    specifications = product.get("specifications")
    specifications = specifications if isinstance(specifications, dict) else {}
    normalized_specs = {str(key).casefold().replace("_", " "): value for key, value in specifications.items()}
    features = product.get("features", [])
    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, (list, tuple)):
        features = []
    row = {
        "shopper_profile": user_context,
        "hard_requirements": user_context,
        "priorities": priority,
        "product_name": str(product.get("name", "")),
        "price": str(product.get("price", "")),
        "price_currency": str(product.get("currency", "")),
        "product_category": infer_model_category(product) or "unknown",
        "product_attributes": "; ".join(
            [f"{key}: {value}" for key, value in sorted(normalized_specs.items())]
            + [str(value) for value in features if isinstance(value, str)]
        ),
    }
    for field, aliases in _SPEC_FIELDS.items():
        value = next((value for key, value in normalized_specs.items()
                      if any(alias in key for alias in aliases)), "")
        row[field] = str(value)
    return row


def recommend_candidates(products: list[dict[str, Any]], priority: str,
                         chat: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Rank supported same-category listings using the small labeled examples.

    Scores are not calibrated probabilities or evidence that a listing meets a
    shopper's requirements. Categories with fewer scenarios have especially
    weak evidence; the app's primary LLM recommendation remains separate.
    """
    if not 2 <= len(products) <= 5:
        raise ValueError("Deality Match requires between two and five product listings.")
    if not supports_deality_match_products(products):
        raise ValueError("Deality Match supports only same-category comparisons with examples in its training data.")

    original_pilot_scenarios = load_labeled_scenarios(DEFAULT_DATA)
    scenarios = load_deality_match_scenarios()
    training_rows = [row for rows in scenarios.values() for row in rows]
    if not scenarios:
        raise ValueError("No eligible labeled scenarios are available for Deality Match.")
    model = make_model()
    model.fit([row_text(row) for row in training_rows], [row["user_preferred"] for row in training_rows])
    candidate_rows = [_candidate_row(product, priority, chat or []) for product in products]
    probabilities = model.predict_proba([row_text(row) for row in candidate_rows])
    positive_column = list(model.classes_).index("yes")
    scores = [float(item[positive_column]) for item in probabilities]
    best_score = max(scores)
    winners = [index for index, score in enumerate(scores) if abs(score - best_score) <= 1e-9]
    selected_category = infer_model_category(products[0])
    category_scenarios = sum(
        any(row.get("product_category") == selected_category for row in rows)
        for rows in scenarios.values()
    )
    combined_matches, _, _, _ = evaluate(scenarios)
    original_pilot_matches, _, _, _ = evaluate(original_pilot_scenarios)
    return {
        "winner_index": winners[0] if len(winners) == 1 else None,
        "training_scenarios": len(scenarios),
        "candidate_count": len(training_rows),
        "category": selected_category,
        "category_scenarios": category_scenarios,
        "combined_cv_matches": combined_matches,
        "combined_cv_scenarios": len(scenarios),
        "original_pilot_matches": original_pilot_matches,
        "original_pilot_scenarios": len(original_pilot_scenarios),
    }


def evaluate(scenarios: dict[str, list[dict[str, str]]]) -> tuple[int, int, list[str], list[str]]:
    """Leave one entire scenario out; return model/baseline wins and details."""
    scenario_ids = list(scenarios)
    model_correct: list[str] = []
    baseline_correct: list[str] = []
    for held_out in scenario_ids:
        training = [row for key, rows in scenarios.items() if key != held_out for row in rows]
        candidates = scenarios[held_out]
        model = make_model()
        model.fit([row_text(row) for row in training], [row["user_preferred"] for row in training])
        probabilities = model.predict_proba([row_text(row) for row in candidates])
        positive_column = list(model.classes_).index("yes")
        selected = max(range(len(candidates)), key=lambda index: probabilities[index][positive_column])
        preferred = next(index for index, row in enumerate(candidates) if row["user_preferred"] == "yes")
        if selected == preferred:
            model_correct.append(held_out)

        def candidate_price(item: tuple[int, dict[str, str]]) -> tuple[float, int]:
            index, row = item
            try:
                amount = float(row.get("price", ""))
            except (TypeError, ValueError):
                amount = float("inf")
            return amount, index

        cheapest = min(enumerate(candidates), key=candidate_price)[0]
        if cheapest == preferred:
            baseline_correct.append(held_out)
    return len(model_correct), len(baseline_correct), model_correct, baseline_correct


def train(data_path: Path = DEFAULT_DATA, model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    scenarios = load_labeled_scenarios(data_path)
    if len(scenarios) < 2:
        raise ValueError("Need at least two scenarios with explicit yes/no winner labels for grouped evaluation.")

    model_wins, baseline_wins, model_correct, baseline_correct = evaluate(scenarios)
    all_rows = [row for rows in scenarios.values() for row in rows]
    final_model = make_model()
    final_model.fit([row_text(row) for row in all_rows], [row["user_preferred"] for row in all_rows])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "text_fields": TEXT_FIELDS,
                 "training_scenarios": list(scenarios), "warning": "Experimental user-labeled pilot; not production-ready."},
                model_path)

    result = {
        "scenarios": len(scenarios), "candidates": len(all_rows),
        "model_top1": model_wins / len(scenarios),
        "cheapest_top1": baseline_wins / len(scenarios),
        "model_correct_scenarios": model_correct,
        "baseline_correct_scenarios": baseline_correct,
        "model_path": model_path,
    }
    print(f"Eligible labeled scenarios: {result['scenarios']}")
    print(f"Explicitly labeled candidate rows: {result['candidates']}")
    print(f"Leave-one-scenario-out top-1: model {model_wins}/{len(scenarios)}; cheapest baseline {baseline_wins}/{len(scenarios)}")
    print("Model selected the labeled preference in:", model_correct or "none")
    print("Cheapest baseline selected the labeled preference in:", baseline_correct or "none")
    print("WARNING: This is a tiny, unverified user-labeled pilot; these scores are not reliable performance estimates.")
    print("The saved artifact is experimental; the app fits an in-memory copy for Deality Match.")
    print("Saved local model artifact:", model_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    arguments = parser.parse_args()
    train(arguments.data, arguments.model)