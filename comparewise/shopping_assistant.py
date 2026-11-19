"""Deterministic answers using only the two supplied shopping listings.

No listing is fetched or verified here. History is caller-owned; only the
latest user message is consulted for a narrow, vague follow-up.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


def _number(value: Any, maximum: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value.replace(",", "") if isinstance(value, str) else value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 and (maximum is None or number <= maximum) else None


def _reviews(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number.is_integer() else None


def _name(product: Mapping[str, Any], index: int) -> str:
    return str(product.get("name") or f"Option {index + 1}")


def _money(product: Mapping[str, Any]) -> str:
    price = _number(product.get("price"))
    if price is None:
        return "price not returned"
    return str(product.get("price_display") or f"{product.get('currency') or 'currency unknown'} {price:,.2f}")


def _summary(products: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for index, product in enumerate(products):
        rating = _number(product.get("rating"), 5)
        reviews = _reviews(product.get("review_count"))
        rating_text = f"{rating:g}/5" if rating is not None else "rating unavailable"
        review_text = f"{reviews:,.0f} reviews" if reviews is not None else "review count unavailable"
        seller = f", seller: {product['seller']}" if product.get("seller") else ""
        parts.append(f"Option {index + 1}, {_name(product, index)}{seller}: {_money(product)}, {rating_text}, {review_text}.")
    return " ".join(parts)


def _price_issue(products: Sequence[Mapping[str, Any]]) -> str | None:
    if any(_number(p.get("price")) is None for p in products):
        return "A price is missing or invalid for at least one listing, so prices cannot be compared."
    currencies = [str(p.get("currency") or "").strip().upper() for p in products]
    if not all(re.fullmatch(r"[A-Z]{3}", currency, re.ASCII) for currency in currencies):
        return "A currency is missing or invalid, so prices cannot be compared."
    if currencies[0] != currencies[1]:
        return "The listings have different currencies; prices cannot be compared without conversion."
    return None


def _price_question_answer(products: Sequence[Mapping[str, Any]]) -> str:
    prices = [_number(product.get("price")) for product in products]
    if any(price is None for price in prices):
        return "The prices cannot be compared because a price wasn't provided for every listing. " + _summary(products)

    issue = _price_issue(products)
    if issue:
        if "different currencies" in issue:
            return f"I can't tell which is cheaper: {issue} " + _summary(products)
        lower = min(range(len(prices)), key=prices.__getitem__)
        if all(price == prices[0] for price in prices):
            return f"The listings show the same numeric price ({_money(products[0])}), but {issue} " + _summary(products)
        others = " and ".join(_money(product) for index, product in enumerate(products) if index != lower)
        return (f"{_name(products[lower], lower)} has the lower displayed amount ({_money(products[lower])} vs {others}). "
            f"The currency code wasn't provided. {issue} I can't confirm these are comparable prices without matching currency codes. " + _summary(products))

    if all(price == prices[0] for price in prices):
        return f"The listings have the same returned price ({_money(products[0])}). " + _summary(products)
    lower = min(range(len(prices)), key=prices.__getitem__)
    others = " and ".join(_money(product) for index, product in enumerate(products) if index != lower)
    return f"{_name(products[lower], lower)} has the lower listed price ({_money(products[lower])} vs {others}). " + _summary(products)


def _contextual_question(prompt: str, history: Sequence[Mapping[str, Any]] | None) -> str:
    vague = re.fullmatch(
        r"(?:and\s+|so\s+|then\s+)?(?:which (?:one )?(?:would|should) (?:you|i) (?:choose|pick|buy|get)|"
        r"which (?:one )?(?:is best|is better|do you (?:recommend|prefer))|"
        r"what (?:would you (?:choose|pick|recommend)|about (?:that|those|them))|"
        r"(?:your )?(?:recommendation|choice|opinion))\s*[?.!]*", prompt, re.I,
    )
    if not vague or not isinstance(history, (list, tuple)):
        return prompt
    skipped = False
    for message in reversed(history):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if not skipped and content.strip().casefold() == prompt.casefold():
            skipped = True
            continue
        return content.strip() + " " + prompt
    return prompt


_CURRENCIES = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR",
               "usd": "USD", "eur": "EUR", "gbp": "GBP", "cad": "CAD",
               "aud": "AUD", "inr": "INR", "dollars": "USD", "dollar": "USD",
               "euros": "EUR", "euro": "EUR", "pounds": "GBP", "pound": "GBP"}
_TOKEN = r"(?:usd|eur|gbp|cad|aud|inr|dollars?|euros?|pounds?|[$€£₹])"
_AMOUNT = r"\d+(?:,\d{3})*(?:\.\d{1,2})?"
_BUDGET = re.compile(
    rf"\b(?P<op>under|below|less than|maximum|max|budget(?:\s+(?:of|is))?|up to)\s*"
    rf"(?P<before>{_TOKEN})?\s*(?P<amount>{_AMOUNT})(?![\d.,])\s*(?P<after>{_TOKEN}|(?!and\b|for\b|the\b)[a-z]{{3}}\b|yen\b|rupees?\b)?(?!\w)", re.I,
)


def _budget(question: str) -> tuple[float, str, bool] | None:
    match = _BUDGET.search(question)
    if not match:
        return None
    before, after = match.group("before"), match.group("after")
    # Conflicting currency markers are ambiguous and cannot support a comparison.
    first = _CURRENCIES.get(before.casefold()) if before else None
    second = _CURRENCIES.get(after.casefold()) if after else None
    currency = "" if (after and not second) or (first and second and first != second) else first or second or "USD"
    return float(match.group("amount").replace(",", "")), currency, match.group("op").lower() in ("under", "below", "less than")


_IGNORED = set("which what does has have with from this that better compare both their show tell about specification specifications product products listing listings would should choose recommend recommendation opinion pick buy need want looking find please one best fits budget under below less than maximum price cost dollars euro euros pounds feature features movement must highest higher lowest lower rating rated reviews review customers customer value only just care now more most cheapest cheaper option options good for and the your".split())


def _feature_evidence(question: str, products: Sequence[Mapping[str, Any]]) -> str:
    text = _BUDGET.sub(" ", question.casefold())
    names = [set(re.findall(r"[a-z]+", str(p.get("name") or "").casefold())) for p in products]
    common = names[0] & names[1]
    terms = {word for word in re.findall(r"[a-z0-9]+", text) if len(word) > 3 and word not in _IGNORED and word not in common}
    generic = bool(re.search(r"\b(?:features?|specs?|specifications?)\b", text))
    specific = bool(re.search(r"\b(?:has|have|with|include|includes|including|must.have|need|needs|want|wants|for)\b", text))
    rows, any_match = [], False
    for index, product in enumerate(products):
        specs = product.get("specifications")
        features = product.get("features")
        fields = [f"{key}: {value}" for key, value in specs.items() if value is not None and str(value).strip()] if isinstance(specs, Mapping) else []
        fields += [str(value) for value in features if value is not None and str(value).strip()] if isinstance(features, (list, tuple)) else []
        if product.get("description"):
            fields.append(str(product["description"]))
        matches = [field for field in fields if any(re.search(r"\b" + re.escape(term) + r"\b", field.casefold()) for term in terms)]
        any_match |= bool(matches)
        if generic and not terms:
            matches = fields
        rows.append(f"Option {index + 1}, {_name(product, index)}: " + ("; ".join(dict.fromkeys(matches)) if matches else "no matching detail was returned"))
    if not (any_match or generic or (specific and terms)):
        return ""
    return "Listing evidence for your requested detail: " + " | ".join(rows) + ". I can’t infer unlisted capabilities or verify suitability from keyword matches."


def _weight_answer(question: str, products: Sequence[Mapping[str, Any]]) -> str:
    values = []
    for product in products:
        details = []
        specs = product.get("specifications")
        if isinstance(specs, Mapping):
            details.extend(str(value).strip() for key, value in specs.items()
                           if "weight" in str(key).casefold() and value is not None and str(value).strip())
        features = product.get("features")
        if isinstance(features, (list, tuple)):
            details.extend(str(value).strip() for value in features
                           if value is not None and re.search(r"\b(?:weight|weighs?)\b", str(value), re.I))
        description = str(product.get("description") or "")
        if description and re.search(r"\bweight|weighs?\b", description, re.I):
            details.append(description)
        values.append("; ".join(dict.fromkeys(details)) or None)

    names = [_name(product, index) for index, product in enumerate(products)]
    if not all(values):
        details = " | ".join(f"{name}: {value or 'weight not provided'}" for name, value in zip(names, values))
        return f"The supplied listings don't include enough weight information to compare them. {details}. Check the product listings for weight details."

    def grams(value: str) -> float | None:
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(kg|kilograms?|g|grams?|lb|lbs|pounds?|oz|ounces?)\b", value, re.I)
        if not match:
            return None
        amount = float(match.group(1))
        unit = match.group(2).casefold()
        factors = {"kg": 1000, "kilogram": 1000, "kilograms": 1000, "g": 1, "gram": 1, "grams": 1,
                   "lb": 453.59237, "lbs": 453.59237, "pound": 453.59237, "pounds": 453.59237,
                   "oz": 28.349523125, "ounce": 28.349523125, "ounces": 28.349523125}
        return amount * factors[unit]

    amounts = [grams(value) for value in values]
    stated = " | ".join(f"{name}: {value}" for name, value in zip(names, values))
    if any(amount is None for amount in amounts):
        return f"The listings report these weight details: {stated}. The units or format aren't clear enough to compare the weights."
    if math.isclose(amounts[0], amounts[1], rel_tol=0.0, abs_tol=1e-9):
        return f"The listings report the same weight: {stated}."
    heavier = 0 if amounts[0] > amounts[1] else 1
    lighter = 1 - heavier
    if re.search(r"\b(?:lighter|lightest)\b", question, re.I):
        return f"{names[lighter]} is lighter based on the listed weights. {stated}."
    if re.search(r"\b(?:heavier|heaviest|heavy)\b", question, re.I):
        return f"{names[heavier]} is heavier based on the listed weights. {stated}."
    return f"The listings report these weights: {stated}."


def _recommendation(products: Sequence[Mapping[str, Any]], priority: str, confirmed: bool) -> str:
    summary = _summary(products)
    try:
        shopping_comparison = importlib.import_module(".shopping_comparison", __package__)
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}.shopping_comparison":
            raise
        return "The shopping comparison module is unavailable; no recommendation can be calculated. " + summary
    if priority in shopping_comparison.SPEC_FOCUSES:
        rows = [row for row in shopping_comparison.comparison_rows(products, confirmed=confirmed)
                if any(term.casefold() in " ".join(str(v) for v in row.values()).casefold()
                       for term in shopping_comparison.SPEC_FOCUSES[priority])]
        evidence = " | ".join(f"{row['Detail']}: Option 1 — {row['Option 1']}; Option 2 — {row['Option 2']}" for row in rows)
        return f"For {priority.lower()}, the supplied evidence cannot establish a winner. " + (evidence or "No matching details were returned.")
    if priority in {"Lower price", "Balanced value", "Price-focused value", "Rating-focused value"}:
        issue = _price_issue(products)
        if issue:
            return "No supported winner for this priority: " + issue + " " + summary
    if priority in {"Higher customer rating", "Balanced value", "Price-focused value", "Rating-focused value"} and any(_number(p.get("rating"), 5) is None for p in products):
        return "No supported winner: a rating is missing or invalid for at least one listing. " + summary
    if priority == "More customer reviews" and any(_reviews(p.get("review_count")) is None for p in products):
        return "No supported winner: a review count is missing or invalid for at least one listing. " + summary
    try:
        result = shopping_comparison.recommend_products(products[0], products[1], priority, confirmed=confirmed)
    except ValueError as error:
        return "I can’t recommend for this comparison: " + str(error) + " " + summary
    winner = result.get("winner_index")
    explanation = str(result.get("explanation") or "No explanation was returned.")
    scores = str(result.get("score_text") or "")
    score_detail = "" if not scores or scores in explanation else " " + scores
    if type(winner) is not int or winner not in (0, 1):
        return f"No supported winner: {explanation}{score_detail} {summary}"
    return f"My comparison-based opinion: {_name(products[winner], winner)} is the pick for {priority.lower()}. {explanation}{score_detail} This is not a guarantee of quality or suitability. {summary}"


def answer_shopping_question(question: str, products: Sequence[Mapping[str, Any]], priority: str = "Balanced value", history: Sequence[Mapping[str, Any]] | None = None, *, confirmed: bool = False) -> str:
    """Answer for exactly two listings; bare amounts and $ mean USD, never conversion.

    Explicit USD, EUR, GBP, CAD, AUD and INR budgets are supported. Listing
    fields are claims from the supplied data, not independently verified facts.
    Set confirmed=True only after the user confirms an uncertain comparison;
    it never overrides a pair classified as incompatible.
    """
    if not isinstance(products, (list, tuple)) or len(products) != 2 or not all(isinstance(p, Mapping) for p in products):
        return "Load exactly two shopping listings first. I can only use details supplied for the current comparison."
    prompt = str(question or "").strip()
    if not prompt:
        return "Tell me your use case, budget, or must-have features and I’ll compare the supplied listing details."
    try:
        comparison = importlib.import_module(".shopping_comparison", __package__)
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}.shopping_comparison":
            raise
        return "The shopping comparison module is unavailable; this comparison cannot be validated. " + _summary(products)
    compatibility = comparison.comparison_compatibility(products[0], products[1])
    if compatibility.get("status") == "incompatible" or (compatibility.get("status") != "compatible" and confirmed is not True):
        return "I can’t compare these listings: " + str(compatibility.get("reason") or "Product compatibility requires confirmation.")
    text = _contextual_question(prompt, history).casefold()
    summary = _summary(products)
    target = next((p for p in products if p.get("name") and str(p["name"]).casefold() in text), None)
    if re.search(r"\b(?:weight|weigh|heavy|heavier|heaviest|light|lighter|lightest)\b", text):
        return _weight_answer(text, products)
    if re.search(r"\b(?:cons|downside|disadvantage|risk|weakness|drawback)s?\b", text):
        selected = [target] if target is not None else products
        rows = []
        for product in selected:
            missing = []
            if _number(product.get("price")) is None:
                missing.append("price was not returned")
            if _number(product.get("rating"), 5) is None:
                missing.append("customer rating was not returned")
            if not (product.get("specifications") or product.get("features")):
                missing.append("specification/feature details were not returned")
            rows.append(f"{product.get('name') or 'Listing'}: " + ("; ".join(missing) if missing else "no clear drawback is established by the supplied fields"))
        return "Comparison limitations: " + " | ".join(rows) + ". I won’t invent downsides; check the listing for additional context."
    budget = _budget(text)
    evidence = _feature_evidence(text, products)
    if budget:
        amount, currency, strict = budget
        issue = _price_issue(products)
        if not currency:
            answer = "The budget currency is ambiguous; please specify one currency."
        elif issue:
            answer = f"For your {currency} budget of {amount:,.2f}, {issue}"
        elif str(products[0].get("currency")).strip().upper() != currency:
            answer = f"The budget is in {currency}, but listing prices use a different currency; conversion is needed."
        else:
            prices = [_number(p.get("price")) for p in products]
            within = [i for i, price in enumerate(prices) if price < amount or (not strict and price == amount)]
            boundary = "under" if strict else "within"
            if not within:
                answer = f"Neither returned price is {boundary} your {currency} budget of {amount:,.2f}."
            elif len(within) == 1:
                i = within[0]
                answer = f"{_name(products[i], i)} is the only listing with a returned price {boundary} your {currency} budget of {amount:,.2f} ({_money(products[i])})."
            elif prices[0] == prices[1]:
                answer = f"Both prices are {boundary} your {currency} budget of {amount:,.2f} and tied; there is no price-based winner."
            else:
                i = 0 if prices[0] < prices[1] else 1
                answer = f"Both prices are {boundary} your {currency} budget of {amount:,.2f}; {_name(products[i], i)} has the lower returned price ({_money(products[i])})."
        return answer + " " + summary + " Prices can change; shipping and other costs are not included." + (" " + evidence + " The budget result alone does not establish a feature match." if evidence else "")
    # An unrecognized currency/amount must not silently become a recommendation.
    if re.search(r"\b(?:under|below|less than|budget|maximum|max|up to)\b", text) and re.search(r"\d", text):
        return "I couldn’t safely interpret that budget and currency; specify an amount and currency code. " + summary + (" " + evidence if evidence else "")
    if re.search(r"\b(?:cheaper|cheapest|lower price|lowest price|price only|which costs less)\b", text):
        return _price_question_answer(products)
    if evidence:
        return evidence
    if (re.search(r"\b(?:recommend|best|worth|opinion|choose|pick|better|compare|cheaper|cheapest)\b", text)
            or re.search(r"\b(?:highest rated|higher rated|highest rating|higher rating)\b", text) or target):
        if re.search(r"\b(?:cheapest|cheaper|lowest price|lower price|price only|only (?:the )?price)\b", text):
            priority = "Lower price"
        elif re.search(r"\b(?:more|most|highest|higher)\s+(?:customer\s+)?reviews\b", text):
            priority = "More customer reviews"
        elif re.search(r"\b(?:highest rated|higher rated|highest rating|higher rating)\b", text):
            priority = "Higher customer rating"
        return _recommendation(products, priority, confirmed)
    if re.search(r"\b(?:rating|ratings|reviews|review|customers)\b", text):
        return "Listing feedback: " + summary + " Ratings and review counts are supplied listing values, not an independent review analysis."
    return "I couldn't find that detail in the supplied listings, so I won't guess. Check both product listings for more information. " + summary