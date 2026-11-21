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


def _named_options(text: str, products: Sequence[Mapping[str, Any]]) -> str:
    for index, product in enumerate(products):
        text = re.sub(rf"\bOption {index + 1}\b", _name(product, index), text)
    return text


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
        parts.append(f"{_name(product, index)}{seller}: {_money(product)}, {rating_text}, {review_text}.")
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


def _price_question_answer(products: Sequence[Mapping[str, Any]], *, more_expensive: bool = False) -> str:
    prices = [_number(product.get("price")) for product in products]
    if any(price is None for price in prices):
        return "I can't compare the prices because a price is missing."

    issue = _price_issue(products)
    if issue and "different currencies" in issue:
        return "I can't compare prices in different currencies."

    if all(price == prices[0] for price in prices):
        return f"The prices are tied at {_money(products[0])}."
    selected = (max if more_expensive else min)(range(len(prices)), key=prices.__getitem__)
    others = " and ".join(_money(product) for index, product in enumerate(products) if index != selected)
    if issue:
        comparison = "higher" if more_expensive else "lower"
        return f"{_name(products[selected], selected)} has the {comparison} displayed price: {_money(products[selected])} vs {others}."
    if more_expensive:
        return f"{_name(products[selected], selected)} is more expensive: {_money(products[selected])} vs {others}."
    return f"{_name(products[selected], selected)} is cheaper: {_money(products[selected])} vs {others}."


def _contextual_question(prompt: str, history: Sequence[Mapping[str, Any]] | None) -> str:
    vague_pattern = re.compile(
        r"(?:and\s+|so\s+|then\s+|like\s+)?(?:which (?:one )?(?:would|should) (?:you|i) (?:choose|pick|buy|get)|"
        r"which (?:one )?(?:is best|is better|do you (?:recommend|prefer))|"
        r"what (?:do you think(?: about (?:that|those|them))?|would you (?:choose|pick|recommend)|about (?:that|those|them))|"
        r"(?:your )?(?:recommendation|choice|opinion)|why)\s*[?.!]*", re.I,
    )
    if not vague_pattern.fullmatch(prompt) or not isinstance(history, (list, tuple)):
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
        if vague_pattern.fullmatch(content.strip()):
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
        rows.append(f"{_name(product, index)}: " + ("; ".join(dict.fromkeys(matches)) if matches else "no matching detail was returned"))
    if not (any_match or generic or (specific and terms)):
        return ""
    return "The details most relevant to your question are: " + " | ".join(rows) + ". I can't verify details that the listings don't provide."


_CAPACITY = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(TB|GB)\b", re.I)
_STORAGE_CONTEXT = re.compile(r"\b(?:storage|ssd|hdd|hard ?drive|disk|rom|flash storage)\b", re.I)
_PHONE_CONTEXT = re.compile(r"\b(?:phone|smartphone|iphone|pixel|galaxy|ipad|tablet)\b", re.I)
_CAMERA_TERMS = re.compile(r"\b(?:camera|cameras|photograph|photography|photos?|selfie|sensor|lens|lenses|megapixels?|zoom|aperture|ois|stabilization|4k|8k|hdr)\b", re.I)
_CAMERA_SPEC_SIGNAL = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mp|megapixels?|x(?:\s+optical)?|k|mm|f/)\b|"
    r"\b(?:telephoto|ultra[ -]?wide|macro|optical zoom|digital zoom|ois|image stabilization|"
    r"sensor.shift|periscope|variable aperture|dual|triple|quad)\b", re.I,
)
_CAMERA_CAPABILITIES = (
    ("telephoto lens", re.compile(r"\btelephoto\b", re.I)),
    ("optical zoom", re.compile(r"\boptical zoom\b", re.I)),
    ("ultra-wide lens", re.compile(r"\bultra[ -]?wide\b", re.I)),
    ("macro lens", re.compile(r"\bmacro\b", re.I)),
    ("image stabilization", re.compile(r"\b(?:ois|optical image stabilization|image stabilization|sensor.shift)\b", re.I)),
    ("periscope lens", re.compile(r"\bperiscope\b", re.I)),
)


def _camera_details(product: Mapping[str, Any]) -> list[str]:
    specs = product.get("specifications")
    candidates = []
    if isinstance(specs, Mapping):
        candidates.extend(f"{key}: {value}" for key, value in specs.items() if value is not None)
    features = product.get("features")
    if isinstance(features, (list, tuple)):
        candidates.extend(str(value) for value in features if value is not None)
    description = product.get("description")
    if isinstance(description, str):
        candidates.extend(re.split(r"(?<=[.!?;])\s+", description))

    details = []
    for candidate in candidates:
        candidate = re.sub(r"\s*\(?\d[\d,]*\s+(?:user\s+)?reviews?\)?", "", str(candidate), flags=re.I).strip(" .;,")
        if (_CAMERA_TERMS.search(candidate) and _CAMERA_SPEC_SIGNAL.search(candidate)
                and candidate.casefold() not in {detail.casefold() for detail in details}):
            details.append(candidate[:240])
    return details[:5]


def _camera_answer(products: Sequence[Mapping[str, Any]]) -> str:
    names = [_name(product, index) for index, product in enumerate(products)]
    details = [_camera_details(product) for product in products]
    detail_text = ["; ".join(items) if items else "no concrete camera specs listed" for items in details]
    evidence = f"{names[0]}: {detail_text[0]} | {names[1]}: {detail_text[1]}"
    if not any(details):
        return (f"I can't fairly call either camera better from these listings. Neither gives concrete camera specs; "
                f"generic camera labels and review counts don't describe camera hardware. {evidence}.")
    if not all(details):
        return (f"I can't fairly rank these cameras because only one listing provides concrete camera specs. "
                f"{evidence}. Check the other product's camera details for a like-for-like comparison.")

    capability_sets = [
        {label for label, pattern in _CAMERA_CAPABILITIES if pattern.search(" ".join(items))}
        for items in details
    ]
    if len(capability_sets[0]) != len(capability_sets[1]):
        winner = 0 if len(capability_sets[0]) > len(capability_sets[1]) else 1
        other = 1 - winner
        added = capability_sets[winner] - capability_sets[other]
        if added:
            reasons = {
                "telephoto lens": "a dedicated telephoto lens can frame distant subjects",
                "optical zoom": "optical zoom can retain more detail at a distance than digital cropping",
                "ultra-wide lens": "an ultra-wide lens can fit more of a scene into the frame",
                "macro lens": "a macro lens is designed for close-up subjects",
                "image stabilization": "stabilization can help reduce blur from hand movement",
                "periscope lens": "a periscope lens supports longer-range zoom designs",
            }
            explanation = "; ".join(reasons[label] for label in sorted(added))
            difference = abs(len(capability_sets[0]) - len(capability_sets[1]))
            return (f"Based on the listed hardware, {names[winner]} has the more versatile camera system on paper: "
                    f"it lists {len(capability_sets[winner])} of these camera capabilities versus "
                    f"{len(capability_sets[other])} for {names[other]} ({difference} more explicitly listed). "
                    f"In particular, {explanation}. {evidence}. These specs suggest more shooting flexibility, "
                    "not guaranteed better-looking photos.")

    return (f"Both listings provide camera details, but they don't establish a clear overall winner. "
            f"{evidence}. Megapixel counts or video resolution alone don't prove which takes better photos.")


def _storage_value(product: Mapping[str, Any]) -> tuple[float, str] | None:
    specs = product.get("specifications")
    candidates = []
    if isinstance(specs, Mapping):
        candidates.extend(str(value) for key, value in specs.items()
                          if _STORAGE_CONTEXT.search(str(key)) and value is not None)
    for field in ("features", "description"):
        value = product.get(field)
        values = value if isinstance(value, (list, tuple)) else (value,)
        candidates.extend(str(item) for item in values if item is not None and _STORAGE_CONTEXT.search(str(item)))
    name = str(product.get("name") or "")
    if _PHONE_CONTEXT.search(name):
        candidates.append(name)
    for candidate in candidates:
        match = _CAPACITY.search(candidate)
        if match:
            amount = float(match.group(1))
            unit = match.group(2).upper()
            gigabytes = amount * 1000 if unit == "TB" else amount
            return gigabytes, f"{amount:g} {unit}"
    return None


def _storage_answer(products: Sequence[Mapping[str, Any]], priority: str, confirmed: bool) -> str:
    names = [_name(product, index) for index, product in enumerate(products)]
    capacities = [_storage_value(product) for product in products]
    details = " | ".join(
        f"{name}: {capacity[1] if capacity else 'storage capacity not listed'}"
        for name, capacity in zip(names, capacities)
    )
    known = [(index, capacity) for index, capacity in enumerate(capacities) if capacity]
    if len(known) == 2:
        if math.isclose(known[0][1][0], known[1][1][0]):
            return f"Both listings show the same storage capacity. {details}. For an overall pick, {_recommendation(products, priority, confirmed)}"
        winner = max(known, key=lambda item: item[1][0])[0]
        other = 1 - winner
        return (f"I'd recommend {names[winner]} if storage is your priority because it lists {capacities[winner][1]}, "
                f"compared with {capacities[other][1]} for {names[other]}. Both are retailer-provided listing details.")
    if len(known) == 1:
        winner, capacity = known[0]
        return (f"I'd tentatively recommend {names[winner]} for storage because its listing states {capacity[1]}. "
                f"{names[1 - winner]}'s capacity isn't listed, so I can't confirm which has more. {details}.")
    fallback_priority = "Balanced value" if priority in {
        "Materials & build", "Battery & power", "Performance & components", "Size & weight", "Warranty & support",
    } else priority
    return (f"Neither listing states its storage capacity, so I can't verify which holds more. "
            f"For an overall pick using the available listing data, {_recommendation(products, fallback_priority, confirmed)} "
            f"Please confirm storage capacity before buying. {details}.")


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


def _gaming_value(product: Mapping[str, Any], kind: str) -> tuple[tuple, str] | None:
    specs = product.get("specifications")
    entries = [(str(key).casefold(), str(value)) for key, value in specs.items()
               if value is not None] if isinstance(specs, Mapping) else []
    if kind == "gpu":
        candidates = [value for key, value in entries if re.search(r"gpu|graphics|video card", key)]
        candidates.extend(str(product.get(field) or "") for field in ("name", "description"))
        features = product.get("features")
        if isinstance(features, (list, tuple)):
            candidates.extend(str(feature) for feature in features if feature is not None)
        for value in candidates:
            nvidia = re.search(r"\b(rtx|gtx)\s*(\d{4})\b", value, re.I)
            if nvidia:
                family, model = nvidia.group(1).casefold(), int(nvidia.group(2))
                return ((2 if family == "rtx" else 1, model // 100, model % 100), value)
            radeon = re.search(r"\brx\s*(\d{4})\b", value, re.I)
            if radeon:
                return ((int(radeon.group(1)) // 100, int(radeon.group(1)) % 100), value)
        return None

    if kind == "cpu":
        candidates = [value for key, value in entries if re.search(r"cpu|processor|chipset", key)]
        candidates.append(str(product.get("name") or ""))
        for value in candidates:
            intel = re.search(r"\b(?:intel\s+)?(?:core\s+)?(?:ultra\s+)?([3579])\s*(\d{3,5})?([a-z]{1,3})?\b", value, re.I)
            if intel and re.search(r"\b(?:intel|core|ultra)\b", value, re.I):
                model = int(intel.group(2) or 0)
                return ((2, int(intel.group(1)), model), value)
            ryzen = re.search(r"\bryzen\s+([3579])\s*(\d{4,5})?([a-z]{1,3})?\b", value, re.I)
            if ryzen:
                return ((1, int(ryzen.group(1)), int(ryzen.group(2) or 0)), value)
        return None

    key_pattern = r"ram|memory" if kind == "ram" else r"refresh rate|display|screen"
    for key, value in entries:
        if not re.search(key_pattern, key):
            continue
        number = re.search(r"\b(\d+(?:\.\d+)?)\s*(gb|hz)\b", value, re.I)
        if number:
            amount = float(number.group(1))
            return ((amount,), value)
    return None


def _gaming_answer(question: str, products: Sequence[Mapping[str, Any]]) -> str:
    if re.search(r"\b(?:gpu|graphics card|graphics|video card)\b", question, re.I):
        kinds = ("gpu",)
    elif re.search(r"\b(?:cpu|processor)\b", question, re.I):
        kinds = ("cpu",)
    elif re.search(r"\b(?:ram|memory)\b", question, re.I):
        kinds = ("ram",)
    elif re.search(r"\b(?:refresh rate|hz|screen speed)\b", question, re.I):
        kinds = ("refresh",)
    else:
        kinds = ("gpu", "cpu", "ram", "refresh")

    names = [_name(product, index) for index, product in enumerate(products)]
    for kind in kinds:
        values = [_gaming_value(product, kind) for product in products]
        available = [(index, data) for index, data in enumerate(values) if data is not None]
        if len(available) < 2:
            continue
        vendors = {data[0][0] for _, data in available} if kind in ("gpu", "cpu") else {"same"}
        if len(available) != len(products) or len(vendors) != 1:
            continue
        highest = max(data[0] for _, data in available)
        leaders = [(index, data) for index, data in available if data[0] == highest]
        if len(leaders) != 1:
            continue
        winner, data = leaders[0]
        label = {"gpu": "listed graphics", "cpu": "listed processor", "ram": "listed memory", "refresh": "listed refresh rate"}[kind]
        return f"For gaming, {names[winner]} is my pick based on its {label} ({data[1]})."

    gpu_values = [_gaming_value(product, "gpu") for product in products]
    known_gpus = [(index, data) for index, data in enumerate(gpu_values) if data is not None]
    gpu_evidence = " | ".join(
        f"{names[index]}: {data[1] if data else 'GPU not listed'}"
        for index, data in enumerate(gpu_values)
    )
    if len(known_gpus) == 1:
        winner, data = known_gpus[0]
        return (f"Based on the only listed GPU, {names[winner]} is the more promising gaming option "
                f"({data[1]}). The other listing's GPU isn't provided, so I can't confirm "
                f"head-to-head performance. {gpu_evidence}")
    if known_gpus:
        return f"For gaming, the available graphics evidence is {gpu_evidence}. I can't reliably rank these GPUs from the supplied listing details."
    return f"I can't reliably compare gaming performance from the supplied listings: GPU details aren't listed for both options. {gpu_evidence}"


def _rating_answer(question: str, products: Sequence[Mapping[str, Any]]) -> str:
    ratings = [_number(product.get("rating"), 5) for product in products]
    names = [_name(product, index) for index, product in enumerate(products)]
    if any(rating is None for rating in ratings):
        details = " | ".join(
            f"{name}: {rating:g}/5" if rating is not None else f"{name}: rating not listed"
            for name, rating in zip(names, ratings)
        )
        return f"I can't compare the customer ratings because a rating is missing. {details}."
    if ratings[0] == ratings[1]:
        return f"The listings show tied customer ratings: {ratings[0]:g}/5 each."
    lower = bool(re.search(r"\b(?:lower|lowest|less|fewer)\b", question, re.I))
    winner = (min if lower else max)(range(len(ratings)), key=ratings.__getitem__)
    other = 1 - winner
    comparison = "lower" if lower else "higher"
    return (f"{names[winner]} has the {comparison} listed customer rating "
            f"({ratings[winner]:g}/5 vs {ratings[other]:g}/5). Ratings are supplied listing values.")


def _recommendation(products: Sequence[Mapping[str, Any]], priority: str, confirmed: bool) -> str:
    summary = _summary(products)
    names = [_name(product, index) for index, product in enumerate(products)]
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
        evidence = " | ".join(f"{row['Detail']}: {names[0]} — {row['Option 1']}; {names[1]} — {row['Option 2']}" for row in rows)
        return f"For {priority.lower()}, I can't identify a supported winner because specifications aren't scored. " + (evidence or "No matching details were returned.")
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
    explanation = _named_options(explanation, products)
    scores = _named_options(str(result.get("score_text") or ""), products)
    if type(winner) is not int or winner not in (0, 1):
        return f"I can't identify a clear winner for {priority.lower()} from these listings. {explanation}"
    score_detail = f" Pair scores: {scores}." if scores and scores not in explanation else ""
    return f"I'd recommend {_name(products[winner], winner)} for {priority.lower()} because {explanation}{score_detail} This is based on the supplied listing values, not a guarantee of quality or suitability."


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
    if (re.search(r"\b(?:ratings?|rated)\b", text)
            and re.search(r"\b(?:higher|highest|lower|lowest|better|more|less|compare|which|what)\b", text)):
        return _rating_answer(text, products)
    if re.search(r"\b(?:camera|cameras|photograph|photography|photos?|selfies?)\b", text):
        return _camera_answer(products)
    if re.search(r"\b(?:storage|capacity|ssd|hard ?drive|disk space|rom)\b", text):
        return _storage_answer(products, priority, confirmed)
    if re.search(r"\b(?:gaming|games|gamer|graphics|gpu|cpu|processor|ram|memory|refresh rate)\b", text):
        return _gaming_answer(text, products)
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
    if re.search(r"\b(?:more expensive|most expensive|pricier|costs? more|higher price|highest price)\b", text):
        return _price_question_answer(products, more_expensive=True)
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