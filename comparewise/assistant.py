"""Grounded deterministic assistant for a current comparison session."""

from __future__ import annotations

import re

from .recommendation import recommend
from .schemas import Product


def _money(amount: float | None, currency: str) -> str:
    return "price not provided" if amount is None else f"{currency} {amount:,.2f}"


def answer_question(
    question: str,
    products: list[Product],
    priorities: dict[str, float] | None = None,
    setup_needs: list[str] | None = None,
    accessories_owned: list[str] | None = None,
    context_product: str | None = None,
) -> str:
    if len(products) < 2:
        return "Add at least two products to start a comparison. I only use details entered or extracted from the product pages."
    text = question.lower()
    effective_priorities = dict(priorities or {})
    preference_changed = False
    priority_terms = {
        "price": ("lowest price", "cheapest", "budget first", "prioritize price", "prioritise price", "care most about price"),
        "performance": ("prioritize performance", "prioritise performance", "focus on performance", "performance first", "gaming first"),
        "quality": ("prioritize quality", "prioritise quality", "quality first", "care most about quality"),
    }
    changed_key = next((key for key, phrases in priority_terms.items() if any(phrase in text for phrase in phrases)), None)
    if changed_key:
        if changed_key == "price" and any(term in text for term in ("lowest price", "cheapest", "budget first")):
            effective_priorities = {key: float(key == "price") for key in (priorities or {})}
            if not effective_priorities:
                effective_priorities = {"price": 1.0}
        else:
            keys = list(priorities or {})
            effective_priorities = {key: float(value) for key, value in (priorities or {}).items()}
            if changed_key not in effective_priorities:
                effective_priorities[changed_key] = 0.7
                keys.append(changed_key)
            current = max(effective_priorities.get(changed_key, 0), 0.7)
            remainder = sum(max(0.0, effective_priorities.get(key, 0)) for key in keys if key != changed_key)
            for key in keys:
                if key == changed_key:
                    effective_priorities[key] = current
                else:
                    effective_priorities[key] = (1 - current) * max(0.0, effective_priorities.get(key, 0)) / remainder if remainder else 0.0
        preference_changed = True

    effective_owned = list(accessories_owned or [])
    if any(phrase in text for phrase in ("already own", "already have", "own a ", "what if")):
        known_items = {item.strip().lower() for p in products for item in p.included_items}
        known_items.update(item.strip().lower() for item in (setup_needs or []))
        effective_owned.extend(
            item for item in known_items
            if item and any(token in text for token in item.split() if len(token) > 2)
            and item not in {x.lower() for x in effective_owned}
        )

    try:
        result = recommend(products, effective_priorities, setup_needs or [], effective_owned)
    except ValueError as error:
        return str(error)
    names = [p.name for p in products]
    selected = next((p for p in products if p.name.lower() in text), None)
    if selected is None:
        selected = next((p for p in products if any(token in text for token in p.name.lower().split() if len(token) > 3)), None)
    chosen_name = selected.name if selected else (context_product if context_product in names else result["best_overall"])
    chosen = next(p for p in products if p.name == chosen_name)
    ranked = result["ranked"]
    chosen_row = next(row for row in ranked if row["name"] == chosen_name)

    if preference_changed:
        ordered = " → ".join(f"{row['name']} ({row['score']:.1f})" for row in ranked)
        label = "lowest price" if changed_key == "price" and effective_priorities.get("price") == 1 else changed_key
        return f"With {label} prioritized, the recalculated ranking is: {ordered}. This is a what-if view for this answer; adjust the priority sliders in the sidebar to save the preference for the whole session."

    if any(word in text for word in ("why", "recommend", "best overall", "winner")):
        unknown = ", ".join(chosen_row["missing_fields"]) or "none of the scored fields"
        top = ", ".join(chosen_row["reasons"][:2])
        return f"{chosen_name} is ranked best overall for the current priorities because {top}. Its score is {chosen_row['score']:.1f}/100 relative to these products. Missing information: {unknown}. This is a model-based comparison, not a guarantee."
    if "gaming" in text or "game" in text:
        candidates = [p for p in products if p.performance is not None]
        if not candidates:
            return "I can't determine gaming performance: no product has a performance score entered. Add verified CPU/GPU details or a sourced benchmark; I won't infer them from the product name."
        best = max(candidates, key=lambda p: p.performance)
        return f"Among products with entered performance scores, {best.name} is highest at {best.performance:.0f}/100. That score's source is not independently verified here; gaming-specific benchmarks and target resolution are missing unless entered in specifications."
    if any(word in text for word in ("disadvantage", "downside", "weakness", "con")):
        fields = []
        if chosen.price is not None and any(p.price is not None and p.price < chosen.price for p in products): fields.append("it costs more than at least one alternative")
        if chosen.performance is None: fields.append("its performance is not provided")
        elif any(p.performance is not None and p.performance > chosen.performance for p in products): fields.append("another product has a higher entered performance score")
        if chosen.quality is None: fields.append("its quality score is not provided")
        if chosen.rating is None: fields.append("no customer rating was supplied")
        if not fields: fields.append("no clear downside can be established from the supplied data; some important fields may still be missing")
        return f"For {chosen_name}, the supported drawbacks are: " + "; ".join(fields) + "."
    if any(word in text for word in ("review", "customer feedback", "what do buyers say")):
        if not chosen.review_notes and chosen.review_count is None and chosen.rating is None:
            return f"No customer ratings or review text are available for {chosen_name}. I can't summarize or infer customer sentiment from the current data."
        details = []
        if chosen.rating is not None: details.append(f"entered rating {chosen.rating:g}/5")
        if chosen.review_count is not None: details.append(f"entered review count {chosen.review_count}")
        if chosen.review_notes: details.append(f"provided review notes: {chosen.review_notes}")
        return f"For {chosen_name}, the supplied review evidence is: " + "; ".join(details) + ". Review notes are user-provided and have not been independently verified."
    if any(word in text for word in ("spec", "specification", "cpu", "gpu", "memory", "ram", "storage")):
        if any(word in text for word in ("which", "better", "compare", "versus", " vs ")):
            common = set(products[0].specifications)
            for product in products[1:]:
                common.intersection_update(product.specifications)
            if not common:
                return "The products do not have common entered specification fields to compare. Add verified CPU, GPU, memory, or storage details; I won't infer missing specifications."
            comparisons = []
            for key in sorted(common):
                values = [(product.name, product.specifications[key]) for product in products if key in product.specifications]
                comparisons.append(f"{key}: " + " vs ".join(f"{name} — {value}" for name, value in values))
            return "Entered specification comparison: " + "; ".join(comparisons) + ". These values may not be directly comparable without model-specific context, so no overall spec winner is assumed."
        if not chosen.specifications:
            return f"No detailed specifications were entered for {chosen_name}. Add verified specification name/value pairs to compare them; I won't fill in missing specs."
        details = "; ".join(f"{key}: {value}" for key, value in chosen.specifications.items())
        return f"Entered specifications for {chosen_name}: {details}. Only those fields are available to compare."
    if "accessor" in text or "already own" in text or "own a" in text or "what if" in text:
        owned = {x.lower() for x in effective_owned}
        needed = {x.lower() for x in (setup_needs or [])} - owned
        if not needed:
            return "No setup needs are currently entered. Add needed items and items you already own; bundle fit will then count only the remaining needs."
        rows = []
        for product in products:
            included = {x.lower() for x in product.included_items}
            covers = sorted(included & needed)
            rows.append(f"{product.name} covers {', '.join(covers) if covers else 'none of the remaining needed accessories'}")
        updated = recommend(products, effective_priorities, setup_needs or [], effective_owned)
        return "Assuming you already own " + ", ".join(sorted(owned) or ["no listed accessories"]) + ", and need " + ", ".join(sorted(needed)) + ": " + "; ".join(rows) + f". For this what-if, the best overall becomes {updated['best_overall']}. Update the setup fields to save the scenario."
    if "worth" in text or "$" in text or "more" in text and "price" in text:
        other = next((p for p in products if p.name != chosen_name and p.price is not None), None)
        if chosen.price is None or other is None:
            return "I can't assess the price difference because a current price is missing for at least one product. Enter verified prices in the same currency."
        delta = abs(chosen.price - other.price)
        expensive = chosen if chosen.price >= other.price else other
        cheaper = other if expensive is chosen else chosen
        return f"{expensive.name} costs {_money(delta, expensive.currency)} more than {cheaper.name} based on entered prices. Whether it is worth it depends on your priorities: current scores are {next(r['score'] for r in ranked if r['name'] == expensive.name):.1f} vs {next(r['score'] for r in ranked if r['name'] == cheaper.name):.1f}. The data does not establish a guaranteed benefit for that price difference."
    if any(word in text for word in ("compare", "which", "better", "versus", " vs ")):
        lines = []
        for row in ranked:
            p = row["product"]
            lines.append(f"{p.name}: {row['score']:.1f}/100; {_money(p.price, p.currency)}; performance {p.performance if p.performance is not None else 'unknown'}/100; quality {p.quality if p.quality is not None else 'unknown'}/100")
        return "Current comparison (higher score reflects your priorities): " + " | ".join(lines) + "."
    return f"I have {len(products)} products in this comparison. Ask about the recommendation, price trade-offs, performance, specifications, drawbacks, or bundle/accessory what-ifs. I can only answer from the entered product data; the current top pick is {result['best_overall']}."
