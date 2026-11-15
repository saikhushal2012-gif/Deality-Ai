"""Train and evaluate Deality Match using grouped product scenarios.

The original laptop pilot, later laptop scenarios, and other-category scenarios
remain separately measurable. Cross-validation holds out whole scenarios.
Small, user-provided and unverified examples do not establish real-world quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score

if __package__ in {None, ""}:  # Support ``python scripts/train_deality_match.py``.
    from train_pilot_model import (
        DEFAULT_ADDITIONAL_DATA, DEFAULT_DATA, DEFAULT_OTHER_DATA, evaluate, make_model, row_text,
        load_additional_scenarios, load_deality_match_scenarios, load_labeled_scenarios,
    )
else:
    from scripts.train_pilot_model import (
        DEFAULT_ADDITIONAL_DATA, DEFAULT_DATA, DEFAULT_OTHER_DATA, evaluate, make_model, row_text,
        load_additional_scenarios, load_deality_match_scenarios, load_labeled_scenarios,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "deality_match.joblib"


def evaluate_grouped(scenarios: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Return out-of-fold candidate-label and scenario top-choice metrics."""
    scenario_ids = list(scenarios)
    actual: list[str] = []
    predicted: list[str] = []
    top1_correct = 0
    for held_out in scenario_ids:
        training = [row for key, rows in scenarios.items() if key != held_out for row in rows]
        candidates = scenarios[held_out]
        model = make_model()
        model.fit([row_text(row) for row in training], [row["user_preferred"] for row in training])
        candidate_text = [row_text(row) for row in candidates]
        fold_predictions = list(model.predict(candidate_text))
        actual.extend(row["user_preferred"] for row in candidates)
        predicted.extend(str(label) for label in fold_predictions)
        probabilities = model.predict_proba(candidate_text)
        yes_column = list(model.classes_).index("yes")
        selected = max(range(len(candidates)), key=lambda index: probabilities[index][yes_column])
        top1_correct += candidates[selected]["user_preferred"] == "yes"
    return {
        "candidate_accuracy": float(accuracy_score(actual, predicted)),
        "top1_correct": int(top1_correct),
        "top1_total": len(scenario_ids),
    }


def train(data_path: Path = DEFAULT_DATA, additional_path: Path = DEFAULT_ADDITIONAL_DATA,
          other_path: Path = DEFAULT_OTHER_DATA,
          model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    original = load_labeled_scenarios(data_path)
    additional_laptops = load_additional_scenarios(additional_path)
    additional_other = load_additional_scenarios(other_path)
    scenarios = load_deality_match_scenarios(data_path, additional_path, other_path)
    duplicate_scenarios = set(original) & (set(additional_laptops) | set(additional_other))
    if duplicate_scenarios:
        raise ValueError(f"Scenario IDs overlap: {', '.join(sorted(duplicate_scenarios))}")
    if set(additional_laptops) & set(additional_other):
        raise ValueError("Scenario IDs overlap between additional data files.")

    train_rows = [row for rows in scenarios.values() for row in rows]
    all_text = [row_text(row) for row in train_rows]
    all_labels = [row["user_preferred"] for row in train_rows]
    model = make_model()
    model.fit(all_text, all_labels)
    training_accuracy = float(accuracy_score(all_labels, model.predict(all_text)))
    cross_validation = evaluate_grouped(scenarios)
    original_correct, _, _, _ = evaluate(original)
    additional_laptop_correct, _, _, _ = evaluate(additional_laptops)
    additional_other_correct, _, _, _ = evaluate(additional_other)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "training_scenarios": sorted(scenarios),
        "original_pilot_scenarios": sorted(original),
        "additional_laptop_scenarios": sorted(additional_laptops),
        "additional_other_scenarios": sorted(additional_other),
        "grouped_candidate_accuracy": cross_validation["candidate_accuracy"],
        "grouped_top1_correct": cross_validation["top1_correct"],
        "grouped_top1_total": cross_validation["top1_total"],
        "warning": "Small, user-provided and unverified laptop pilot; not production-ready.",
    }, model_path)

    result = {
        "scenario_count": len(scenarios),
        "candidate_count": len(train_rows),
        "training_accuracy": training_accuracy,
        "grouped_candidate_accuracy": cross_validation["candidate_accuracy"],
        "grouped_top1_correct": cross_validation["top1_correct"],
        "grouped_top1_total": cross_validation["top1_total"],
        "original_pilot_top1_correct": original_correct,
        "original_pilot_scenarios": len(original),
        "additional_laptop_top1_correct": additional_laptop_correct,
        "additional_laptop_scenarios": len(additional_laptops),
        "additional_other_top1_correct": additional_other_correct,
        "additional_other_scenarios": len(additional_other),
        "model_path": model_path,
    }
    print(f"Deality Match data: {result['candidate_count']} candidates in {result['scenario_count']} scenarios.")
    print(f"Fit-on-all candidate-label accuracy (training, resubstitution): {training_accuracy:.3f}.")
    print(f"Leave-one-scenario-out candidate-label accuracy: {cross_validation['candidate_accuracy']:.3f}.")
    print(f"Leave-one-scenario-out top choice: {cross_validation['top1_correct']}/{cross_validation['top1_total']} scenarios.")
    print(f"Original pilot, evaluated separately: {original_correct}/{len(original)} top choices.")
    print(f"Added laptop scenarios, evaluated separately: {additional_laptop_correct}/{len(additional_laptops)} top choices.")
    print(f"Added other-category scenarios, evaluated separately: {additional_other_correct}/{len(additional_other)} top choices.")
    print("The original pilot metrics remain separate. New listing details are user-provided and unverified.")
    print("Training accuracy is resubstitution and is not an estimate of generalization. Scenario CV is tiny and does not establish real-world quality.")
    print("Saved local model artifact:", model_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--additional-data", type=Path, default=DEFAULT_ADDITIONAL_DATA)
    parser.add_argument("--other-data", type=Path, default=DEFAULT_OTHER_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    train(args.pilot_data, args.additional_data, args.other_data, args.model)
