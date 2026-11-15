"""Train a local preference classifier on the provided synthetic examples.

Examples are converted to the app's labeled-candidate row schema in memory.
Whole scenarios are held out for validation; no held-out row is used to fit
this model. These synthetic metrics do not establish real-world performance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score

if __package__ in {None, ""}:  # Support ``python scripts/train_synthetic_model.py``.
    from train_pilot_model import make_model, row_text
else:
    from scripts.train_pilot_model import make_model, row_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_DATA = ROOT / "data" / "synthetic_preference_examples.json"
DEFAULT_MODEL = ROOT / "artifacts" / "synthetic_preference_model.joblib"
# One entire scenario per category is held out. The same shopper scenario can
# never contribute candidate rows to both the training and validation sets.
VALIDATION_SCENARIO_IDS = {
    "synthetic_laptop_005",
    "synthetic_watch_010",
    "synthetic_hoodie_015",
    "synthetic_musical_instrument_020",
}


def _display_attribute(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if key == "ram" and isinstance(value, (int, float)):
        return f"{value:g} GB"
    if key == "storage" and isinstance(value, (int, float)):
        return f"{value:g} GB"
    if key == "battery_hours":
        return f"{value} hours"
    if key == "battery_days":
        return f"{value} days"
    if key == "weight" and isinstance(value, (int, float)):
        return f"{value:g} kg"
    return str(value)


def convert_examples_to_rows(examples_path: Path = DEFAULT_SYNTHETIC_DATA) -> list[dict[str, str]]:
    """Map the supplied scenario examples to the existing labeled-row schema."""
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        raise ValueError("Synthetic examples must be a JSON list.")

    rows: list[dict[str, str]] = []
    for example in examples:
        category = str(example.get("category", "")).strip()
        scenario_id = f"synthetic_{category}_{int(example['id']):03d}"
        products = example.get("products")
        if not category or not isinstance(products, list) or len(products) < 2:
            raise ValueError(f"Scenario {scenario_id} must include a category and at least two products.")
        product_names = [str(product.get("name", "")).strip() for product in products]
        if not all(product_names) or product_names.count(example.get("preferred")) != 1:
            raise ValueError(f"Scenario {scenario_id} must have exactly one product matching its preferred name.")

        for product in products:
            attributes = {key: value for key, value in product.items() if key not in {"name", "price"}}
            def value_for(*keys: str) -> str:
                for key in keys:
                    if key in attributes:
                        return _display_attribute(key, attributes[key])
                return ""

            rows.append({
                "scenario_id": scenario_id,
                "shopper_profile": str(example.get("answer", "")),
                "budget_max": str(example.get("budget", "")),
                "budget_currency": "USD",
                "hard_requirements": str(example.get("question", "")),
                "priorities": "; ".join(str(item) for item in example.get("priorities", [])),
                "product_name": str(product["name"]),
                "price": str(product.get("price", "")),
                "price_currency": "USD",
                "ram": value_for("ram"),
                "cpu": value_for("cpu"),
                "gpu": value_for("gpu"),
                "storage": value_for("storage"),
                "display": value_for("display"),
                "battery_life": value_for("battery_hours", "battery_days"),
                "weight": value_for("weight"),
                "source_retailer": "Synthetic example",
                "source_note": "User-provided synthetic training example; not a verified real listing.",
                "meets_hard_needs": "",
                "user_preferred": "yes" if product["name"] == example["preferred"] else "no",
                "preference_reason": str(example.get("reason", "")),
                "review_status": "synthetic_user_supplied",
                # Backward-compatible schema extensions for non-laptop examples
                # and category-aware training; old CSV readers ignore these keys.
                "product_category": category,
                "product_attributes": "; ".join(
                    f"{key}: {_display_attribute(key, value)}" for key, value in sorted(attributes.items())
                ),
            })
    return rows


def group_scenarios(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario_id"]].append(row)
    for scenario_id, candidates in grouped.items():
        labels = [row["user_preferred"] for row in candidates]
        if labels.count("yes") != 1 or labels.count("no") < 1:
            raise ValueError(f"{scenario_id} must have exactly one preferred candidate and at least one other.")
    return dict(grouped)


def _top_choice_accuracy(model, scenarios: dict[str, list[dict[str, str]]]) -> tuple[int, int]:
    correct = 0
    for candidates in scenarios.values():
        probabilities = model.predict_proba([row_text(row) for row in candidates])
        yes_column = list(model.classes_).index("yes")
        predicted = max(range(len(candidates)), key=lambda index: probabilities[index][yes_column])
        correct += candidates[predicted]["user_preferred"] == "yes"
    return int(correct), len(scenarios)


def train_synthetic_model(
    data_path: Path = DEFAULT_SYNTHETIC_DATA,
    model_path: Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    scenarios = group_scenarios(convert_examples_to_rows(data_path))
    missing = VALIDATION_SCENARIO_IDS - scenarios.keys()
    if missing:
        raise ValueError(f"Validation scenarios are missing: {', '.join(sorted(missing))}")
    train_scenarios = {key: values for key, values in scenarios.items() if key not in VALIDATION_SCENARIO_IDS}
    validation_scenarios = {key: scenarios[key] for key in sorted(VALIDATION_SCENARIO_IDS)}
    if not train_scenarios:
        raise ValueError("At least one scenario must remain for training.")

    train_rows = [row for candidates in train_scenarios.values() for row in candidates]
    validation_rows = [row for candidates in validation_scenarios.values() for row in candidates]
    x_train = [row_text(row) for row in train_rows]
    y_train = [row["user_preferred"] for row in train_rows]
    x_validation = [row_text(row) for row in validation_rows]
    y_validation = [row["user_preferred"] for row in validation_rows]

    model = make_model()
    model.fit(x_train, y_train)
    training_accuracy = float(accuracy_score(y_train, model.predict(x_train)))
    validation_accuracy = float(accuracy_score(y_validation, model.predict(x_validation)))
    training_top1 = _top_choice_accuracy(model, train_scenarios)
    validation_top1 = _top_choice_accuracy(model, validation_scenarios)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "training_scenarios": sorted(train_scenarios),
        "validation_scenarios": sorted(validation_scenarios),
        "validation_accuracy": validation_accuracy,
        "data_source": "user-provided synthetic scenarios; not real-world evidence",
    }, model_path)

    result = {
        "scenario_count": len(scenarios),
        "training_scenarios": len(train_scenarios),
        "validation_scenarios": len(validation_scenarios),
        "training_candidates": len(train_rows),
        "validation_candidates": len(validation_rows),
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "training_top1": training_top1[0] / training_top1[1],
        "validation_top1": validation_top1[0] / validation_top1[1],
        "validation_scenario_ids": sorted(validation_scenarios),
        "model_path": model_path,
    }
    print("Dataset: user-provided synthetic scenarios only; existing pilot is not included.")
    print(f"Scenario split: {result['training_scenarios']} train / {result['validation_scenarios']} validation; no scenario overlap.")
    print(f"Candidate rows: {result['training_candidates']} train / {result['validation_candidates']} validation.")
    print(f"Candidate-label accuracy: train {training_accuracy:.3f}; validation {validation_accuracy:.3f}.")
    print(f"Scenario top-choice accuracy: train {training_top1[0]}/{training_top1[1]}; validation {validation_top1[0]}/{validation_top1[1]}.")
    print("Input preferences: shopper answer/profile, stated question, budget and within/over-budget signal, priority words, product category/name, and listed product attributes.")
    print("The preferred label is the target; the explanation/reason is excluded from model inputs to prevent label leakage.")
    print("WARNING: These are synthetic examples and do not demonstrate real-world recommendation accuracy.")
    print("Saved separate synthetic model artifact:", model_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_SYNTHETIC_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    train_synthetic_model(args.data, args.model)
