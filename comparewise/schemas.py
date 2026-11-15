"""Typed product schema with explicit missing-data representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


_PRODUCT_TYPE_ALIASES = {
    "watch": "watches", "watches": "watches", "smartwatch": "watches", "smartwatches": "watches",
    "computer": "computers", "computers": "computers", "desktop": "computers",
    "desktop computer": "computers", "desktop computers": "computers",
    "pc": "computers", "pcs": "computers", "gaming pc": "computers", "gaming pcs": "computers",
    "laptop": "laptops", "laptops": "laptops", "notebook": "laptops", "notebooks": "laptops",
    "laptop computer": "laptops", "smartphone": "smartphones", "smartphones": "smartphones",
    "phone": "smartphones", "phones": "smartphones", "mobile phone": "smartphones",
    "cell phone": "smartphones", "cellphone": "smartphones",
    "tablet": "tablets", "tablets": "tablets", "gaming console": "gaming consoles",
    "gaming consoles": "gaming consoles", "television": "televisions", "televisions": "televisions",
    "tv": "televisions", "headphone": "headphones", "headphones": "headphones",
    "earbud": "headphones", "earbuds": "headphones", "earphones": "headphones", "headset": "headphones",
    "camera": "cameras", "cameras": "cameras", "kitchen appliance": "kitchen appliances",
    "kitchen appliances": "kitchen appliances", "laundry appliance": "laundry appliances",
    "laundry appliances": "laundry appliances", "musical instrument": "musical instruments",
    "musical instruments": "musical instruments", "guitar": "musical instruments",
    "clothing / hoodie": "clothing / hoodies", "clothing / hoodies": "clothing / hoodies",
    "hoodie": "clothing / hoodies", "hoodies": "clothing / hoodies", "sweatshirt": "clothing / hoodies",
    "shoe": "shoes", "shoes": "shoes", "sporting good": "sporting goods", "sporting goods": "sporting goods",
}
_PRODUCT_NAME_HINTS = {
    "watches": r"\b(?:smartwatch|smart watch|watch|watches)\b",
    "computers": r"\b(?:desktop pc|gaming pc|computer|desktop|tower pc|pcs?)\b",
    "laptops": r"\b(?:laptop|notebook|macbook)\b",
    "smartphones": r"\b(?:smartphone|mobile phone|cell ?phone|iphone|pixel \d|galaxy s\d)\b",
    "tablets": r"\b(?:tablet|ipad|galaxy tab)\b",
    "gaming consoles": r"\b(?:playstation|xbox|nintendo switch|gaming console)\b",
    "televisions": r"\b(?:television|smart tv|oled tv|4k tv)\b",
    "headphones": r"\b(?:headphones?|headsets?|earbuds?|earphones?)\b",
    "cameras": r"\b(?:camera|camcorder)\b",
    "kitchen appliances": r"\b(?:blender|toaster|air fryer|coffee maker|kettle|microwave|oven)\b",
    "laundry appliances": r"\b(?:washing machine|washer|dryer|laundry appliance)\b",
    "musical instruments": r"\b(?:guitar|piano|keyboard instrument|drum kit|violin|ukulele|musical instrument)\b",
    "clothing / hoodies": r"\b(?:hoodie|hoodies|sweatshirt)\b",
    "shoes": r"\b(?:shoes?|sneakers?|boots?)\b",
}


def normalize_product_type(category: str) -> str:
    """Normalize common singular/plural names without conflating unlike products."""
    value = re.sub(r"[_-]+", " ", str(category or "").casefold())
    value = " ".join(value.split())
    if value in _PRODUCT_TYPE_ALIASES:
        return _PRODUCT_TYPE_ALIASES[value]
    if value.endswith("s") and value[:-1] in _PRODUCT_TYPE_ALIASES:
        return _PRODUCT_TYPE_ALIASES[value[:-1]]
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def infer_product_type(text: str) -> str | None:
    """Detect obvious product-type words for an extra mixed-item safety check."""
    value = str(text or "").casefold()
    matches = [category for category, pattern in _PRODUCT_NAME_HINTS.items() if re.search(pattern, value)]
    return matches[0] if len(matches) == 1 else None


@dataclass
class Product:
    name: str
    category: str
    price: float | None = None
    currency: str = "USD"
    brand: str = ""
    performance: float | None = None  # user-entered 0–100 estimate or sourced score
    quality: float | None = None  # user-entered 0–100 estimate or sourced score
    rating: float | None = None  # 0–5, only when a source is provided
    review_count: int | None = None
    review_notes: str = ""  # manually entered, sourced review evidence; never model-generated
    features_score: float | None = None  # user-entered 0–100 usefulness score
    included_items: list[str] = field(default_factory=list)
    specifications: dict[str, str] = field(default_factory=dict)
    image_url: str = ""
    description: str = ""
    url: str = ""
    source_note: str = ""

    @property
    def is_bundle(self) -> bool:
        return bool(self.included_items)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Product":
        allowed = cls.__dataclass_fields__.keys()
        values = {key: value for key, value in data.items() if key in allowed}
        values["included_items"] = list(values.get("included_items") or [])
        values["specifications"] = dict(values.get("specifications") or {})
        return cls(**values)
