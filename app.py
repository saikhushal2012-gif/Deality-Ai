from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from comparewise import gemini_assistant
from comparewise.shopping_comparison import (
    PRIORITIES, SPEC_FOCUSES, comparison_compatibility, comparison_rows, recommend_many,
)
from comparewise.shopping_assistant import answer_shopping_question
from comparewise.sources import SerpApiError, _http_url, _retailer_url, _shopping_url, search_shopping_products

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
st.set_page_config(page_title="Deality AI", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { color-scheme: dark; --bg:#080b14; --panel:#111827; --line:#29344c; --blue:#79aaff; --cyan:#5be3ee; --purple:#b99cff; --pink:#eea7dc; --muted:#a8b4ca; }
html,body,[class*="css"] {font-family:Inter,"Segoe UI",ui-sans-serif,system-ui,sans-serif;}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 12% 0%,rgba(82,120,255,.16),transparent 35%),radial-gradient(ellipse at 88% 12%,rgba(180,111,255,.13),transparent 32%),var(--bg);color:#f4f6ff;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1280px;padding-top:1.3rem;padding-bottom:3rem;}
h1,h2,h3 {color:#f4f6ff;letter-spacing:-.035em;}
.eyebrow {color:#a7c7ff;font-size:.74rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;}
.brand {display:flex;align-items:center;gap:.65rem;margin:.5rem 0 1.3rem;font-size:1.1rem;font-weight:750;letter-spacing:-.025em;}
.brand-mark {display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;border-radius:.75rem;background:linear-gradient(135deg,#74b7ff,#a485ff 62%,#eea7dc);color:#101327;box-shadow:0 0 24px rgba(134,157,255,.3);}
.hero {position:relative;overflow:hidden;padding:clamp(2.2rem,7vw,5rem) 1.4rem;border:1px solid rgba(151,173,255,.24);border-radius:28px;background:linear-gradient(135deg,rgba(49,83,147,.28),rgba(92,59,139,.22) 53%,rgba(28,114,142,.15)),rgba(15,21,38,.85);text-align:center;box-shadow:0 24px 80px rgba(24,30,68,.28);}
.hero:before {content:"";position:absolute;inset:auto -12% -70% 22%;height:100%;background:radial-gradient(ellipse,rgba(122,158,255,.2),transparent 65%);pointer-events:none;}
.hero h1 {position:relative;margin:.9rem auto;max-width:820px;font-size:clamp(2.35rem,6vw,4.4rem);line-height:1.04;}
.hero h1 span,.price {background:linear-gradient(100deg,#9ac8ff,#b69aff 55%,#f0a9dc);background-clip:text;-webkit-background-clip:text;color:transparent;}
.hero p,.muted {max-width:720px;margin:.7rem auto;color:var(--muted);line-height:1.7;}
.how-card {height:100%;padding:1rem 1.1rem;border:1px solid rgba(151,173,255,.16);border-radius:18px;background:rgba(17,24,39,.72);}
.step-number {color:var(--cyan);font-weight:800;font-size:.8rem;letter-spacing:.12em;}
.product-title {font-size:1.08rem;line-height:1.4;font-weight:700;color:#f4f6ff;overflow-wrap:anywhere;}
.product-image {display:flex;align-items:center;justify-content:center;height:220px;border-radius:16px;background:#f6f7fb;margin:.85rem 0;padding:16px;overflow:hidden;}
.product-image img {width:100%;height:100%;object-fit:contain;}
.product-image.empty {background:rgba(255,255,255,.03);border:1px dashed #485573;color:#a8b4ca;}
.price {font-size:1.22rem;font-weight:800;}
.recommendation,.notice {padding:1.05rem 1.2rem;border:1px solid rgba(151,173,255,.2);border-left:3px solid var(--cyan);border-radius:16px;background:linear-gradient(120deg,rgba(70,102,181,.18),rgba(76,51,119,.18)),rgba(17,24,39,.9);color:#edf1ff;}
.recommendation h3 {margin:.45rem 0;background:linear-gradient(100deg,#9ac8ff,#c09dff);background-clip:text;-webkit-background-clip:text;color:transparent;}
.notice-error {border-left-color:#ff9cbd;}
.stButton > button,.stFormSubmitButton > button {min-height:2.6rem;border-radius:12px;border:1px solid rgba(151,173,255,.28) !important;background:rgba(30,40,64,.78) !important;color:#f4f6ff !important;transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;}
.stButton > button:hover,.stFormSubmitButton > button:hover {transform:translateY(-1px);border-color:rgba(137,185,255,.7) !important;box-shadow:0 8px 26px rgba(94,117,220,.18);}
.stButton > button[data-testid="stBaseButton-primary"],.stFormSubmitButton > button[data-testid="stBaseButton-primary"] {background:linear-gradient(105deg,#8ecaff,#b59aff 58%,#e8a9e3) !important;border:0 !important;color:#101327 !important;box-shadow:0 8px 24px rgba(126,149,255,.24);}
.stButton > button[data-testid="stBaseButton-primary"] p,.stFormSubmitButton > button[data-testid="stBaseButton-primary"] p {color:#101327 !important;font-weight:700;}
input,textarea,[data-baseweb="select"] > div {background:#131a2a !important;color:#f4f6ff !important;border-color:#384360 !important;}
[data-testid="stExpander"] details,[data-testid="stChatMessage"] {background:#111827 !important;border-color:#29344c !important;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:rgba(151,173,255,.2) !important;border-radius:16px !important;}
[data-testid="stDataFrame"] {border:1px solid rgba(151,173,255,.18);border-radius:14px;overflow:hidden;}
.footer {margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(151,173,255,.14);color:#94a1ba;font-size:.82rem;text-align:center;}
@media(max-width:700px) { .block-container {padding:1rem .85rem 2rem;} .hero {border-radius:20px;} .product-image {height:180px;} .stButton > button,.stFormSubmitButton > button {width:100%;} }
</style>
""", unsafe_allow_html=True)

for key, default in (
    ("page", "home"), ("comparison", None), ("shopping_cache", {}),
    *((f"results_{index}", []) for index in range(1, 6)),
    *((f"manual_{index}", None) for index in range(1, 6)),
    ("shopping_chat", []),
    ("current_pair", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


def _notice(message: str, kind: str = "info") -> None:
    st.markdown(f'<div class="notice notice-{kind}">{escape(str(message))}</div>', unsafe_allow_html=True)


def _invalidate() -> None:
    st.session_state.comparison = None
    st.session_state.shopping_chat = []


def _decision(products: list[dict], summary: dict, priority: str) -> tuple[int, str, str]:
    winner = summary["winner_index"]
    if winner is not None:
        return winner, "Recommended pick", summary["explanation"]

    primary = []
    if priority in PRIORITIES:
        primary = sorted(PRIORITIES[priority], key=PRIORITIES[priority].get, reverse=True)
    dimensions = {
        "price": ("price", "Lower listed price"),
        "rating": ("rating", "Higher customer rating"),
        "reviews": ("review_count", "More listed customer reviews"),
    }
    order = [dimensions[key] for key in primary if key in dimensions]
    order.extend(item for item in (dimensions["rating"], dimensions["reviews"], dimensions["price"])
                 if item not in order)

    def comparable_values(field: str) -> list[float | None]:
        values = []
        for product in products:
            value = product.get(field)
            try:
                number = float(value.replace(",", "") if isinstance(value, str) else value)
            except (TypeError, ValueError, OverflowError):
                number = math.nan
            valid = math.isfinite(number) and number >= 0
            if field == "rating":
                valid = valid and number <= 5
            if field == "review_count":
                valid = valid and number.is_integer()
            values.append(number if valid else None)
        if field == "price":
            currencies = {str(product.get("currency") or "").strip().upper() for product in products}
            if len(currencies) != 1 or not next(iter(currencies)):
                return [None] * len(products)
        return values

    for field, label in order:
        values = comparable_values(field)
        available = [(index, value) for index, value in enumerate(values) if value is not None]
        if not available:
            continue
        best = min(value for _, value in available) if field == "price" else max(value for _, value in available)
        leaders = [index for index, value in available if value == best]
        if len(leaders) == 1:
            return leaders[0], "Recommended pick", f"Selected by {label.lower()} in the available listing data."

    return 0, "Recommended pick", "The selected product is first among the compared listings."


def _gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


_RULE_QUESTION = re.compile(
    r"\b(?:cheaper|cheapest|more expensive|most expensive|pricier|costs? more|costs? less|"
    r"lower price|lowest price|higher price|highest price|price only|budget|under|below|"
    r"less than|maximum|max|up to|weight|weigh|heavy|heavier|heaviest|light|lighter|lightest)\b",
    re.I,
)


def _clear() -> None:
    _invalidate()
    st.session_state.shopping_cache = {}
    for key in (f"results_{index}" for index in range(1, 6)):
        st.session_state[key] = []
    for key in (f"manual_{index}" for index in range(1, 6)):
        st.session_state[key] = None
    for key in (f"query_{index}" for index in range(1, 6)):
        st.session_state[key] = ""
    for index in range(1, 6):
        for field in ("name", "category", "seller", "url", "price", "currency", "rating", "reviews", "specs"):
            st.session_state.pop(f"manual_{field}_{index}", None)
        st.session_state.pop(f"input_mode_{index}", None)
    for key in (*(f"selection_{index}" for index in range(1, 6)), "pair_confirmation", "group_confirmation"):
        st.session_state.pop(key, None)
    st.session_state.current_pair = None


def _identity(product: dict) -> str:
    return str(product.get("id") or "")


def _same_result(first: dict, second: dict) -> bool:
    first_id, second_id = _identity(first), _identity(second)
    first_url, second_url = _retailer_url(first.get("url")), _retailer_url(second.get("url"))
    return bool((first_id and first_id == second_id) or (first_url and first_url == second_url) or first == second)


def _format_result(product: dict) -> str:
    return " · ".join((str(product.get("name") or "Unnamed listing"),
                       str(product.get("seller") or "Seller unavailable"),
                       str(product.get("price_display") or "Price unavailable")))


def _render_product(product: dict, index: int) -> None:
    name = str(product.get("name") or "Unnamed listing")
    st.markdown(f'<div class="eyebrow">OPTION {index}</div><div class="product-title">{escape(name)}</div>', unsafe_allow_html=True)
    st.caption(f"Seller: {product.get('seller') or 'Unavailable'}")
    image = _http_url(product.get("image_url"))
    if image:
        st.markdown(f'<div class="product-image"><img src="{escape(image, quote=True)}" alt="{escape(name, quote=True)}" loading="lazy" /></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="product-image empty">Image unavailable</div>', unsafe_allow_html=True)
    display_price = product.get("price_display")
    if not display_price and product.get("price") is not None:
        display_price = f'{product["price"]:g} {product.get("currency") or "(currency unavailable)"}'
    st.markdown(f'<div class="price">{escape(str(display_price or "Price unavailable"))}</div>', unsafe_allow_html=True)
    rating, reviews = st.columns(2)
    with rating:
        st.metric("Rating", f'{product["rating"]:g} / 5' if product.get("rating") is not None else "Not listed")
    with reviews:
        st.metric("Reviews", f'{product["review_count"]:,}' if product.get("review_count") is not None else "Not listed")
    url = _retailer_url(product.get("url"))
    if url:
        st.link_button("View retailer listing ↗", url, width="stretch")
    else:
        st.caption("Retailer link unavailable")
        shopping_url = _shopping_url(product.get("shopping_url"))
        if shopping_url:
            st.link_button("View Google Shopping result ↗", shopping_url, width="stretch")
    if product.get("description"):
        st.write(product["description"])
    features = product.get("features") or []
    with st.expander("Listing highlights"):
        if features:
            for feature in features:
                st.write("• " + str(feature))
        else:
            st.caption("Not listed")
    if product.get("source_note"):
        st.caption(product["source_note"])


def render_home() -> None:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span><span>Deality AI</span></div>', unsafe_allow_html=True)
    st.markdown('''<div class="hero"><div class="eyebrow">A clearer way to shop</div><h1>Find the right value.<br><span>Not just the lowest price.</span></h1><p>Compare retailer offers, check the details that matter to you, and get a recommendation grounded in the listing evidence.</p></div>''', unsafe_allow_html=True)
    st.write("")
    steps = st.columns(3)
    for column, number, title, copy in zip(steps, ("01", "02", "03"),
                                           ("Search", "Compare", "Decide"),
                                           ("Find offers from multiple retailers.", "Review prices and the details actually provided.", "See the trade-offs and choose with more clarity.")):
        with column:
            st.markdown(f'<div class="how-card"><div class="step-number">{number}</div><h3>{title}</h3><p class="muted">{copy}</p></div>', unsafe_allow_html=True)
    st.write("")
    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button("Start comparing →", type="primary", width="stretch"):
            st.session_state.page = "compare"
            st.rerun()
    st.caption("Retailer-independent Google Shopping results · Listing details are not independently verified · Missing information stays unavailable")
    st.markdown('<div class="footer">This project is a prototype and is not financial advice.<br>Need help? Call <a href="tel:+16097514039">609-751-4039</a>.</div>', unsafe_allow_html=True)


def _rule_based_answer(question: str, products: list[dict], priority: str,
                       history: list[dict], confirmed: bool) -> str:
    if len(products) == 2:
        answer = answer_shopping_question(question, products, priority, history, confirmed=confirmed)
    else:
        result = recommend_many(products, priority, confirmed=confirmed)
        winner, label, explanation = _decision(products, result, priority)
        name = products[winner].get("name") or f"Option {winner + 1}"
        answer = f"{label} for {priority.lower()}: {name}. {explanation}"
        answer += " Listing details and statistics appear in the comparison below."
    general_recommendation = bool(re.search(r"\b(?:recommend|recommendation|best|better|opinion|choose|pick)\b", question, re.I)) and not bool(
        re.search(r"\b(?:price|cheaper|cheapest|budget|under|below|less than|maximum|max|up to|rating|reviews|weight|heavier|lighter)\b", question, re.I)
    )
    if len(products) == 2 and general_recommendation and any(
        phrase in answer.casefold()
        for phrase in ("no supported winner", "cannot establish a winner", "no winner:", "scores are tied")
    ):
        result = recommend_many(products, priority, confirmed=confirmed)
        winner, label, explanation = _decision(products, result, priority)
        answer = f"{label} for {priority.lower()}: {products[winner].get('name') or f'Option {winner + 1}'}. {explanation} Listing details weren't provided or comparable for every part of this question."
    return answer


def render_assistant(priority: str) -> None:
    comparison = st.session_state.comparison
    products = comparison["products"] if comparison else []
    confirmed = bool(comparison and comparison["confirmed"])
    api_key = _gemini_api_key()
    st.subheader("Ask Deality AI")
    if api_key:
        st.caption("Gemini handles open-ended questions; your question and selected listing details are sent to Google AI.")
    else:
        st.caption("Set GEMINI_API_KEY to enable online answers. Listing-rule answers work without it.")
    with st.form("shopping_assistant_chat", clear_on_submit=True):
        prompt = st.text_input("Your requirements", placeholder="e.g. Which is cheaper, or what fits my budget?")
        send = st.form_submit_button("Ask Deality AI", type="primary", width="stretch")
    if send:
        question = prompt.strip()
        if not question:
            _notice("Enter a question about the selected listings.", "error")
        elif not 2 <= len(products) <= 5:
            _notice("Compare the selected products first.", "error")
        else:
            try:
                if api_key and not _RULE_QUESTION.search(question):
                    with st.spinner("Thinking through your question..."):
                        answer = gemini_assistant.request_model_answer(
                            question, products, priority, st.session_state.shopping_chat,
                            confirmed=confirmed, api_key=api_key,
                        )
                else:
                    answer = _rule_based_answer(question, products, priority,
                                                st.session_state.shopping_chat, confirmed)
            except (gemini_assistant.ModelAssistantError, ValueError):
                answer = _rule_based_answer(question, products, priority,
                                            st.session_state.shopping_chat, confirmed)
            st.session_state.shopping_chat.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
    with st.container(height=220, border=True):
        if not st.session_state.shopping_chat:
            st.caption("Ask about price, ratings, specifications, or the current recommendation.")
        for message_index, message in enumerate(st.session_state.shopping_chat):
            with st.chat_message(message["role"]):
                st.text(message["content"])
                previous_question = next((entry["content"] for entry in reversed(st.session_state.shopping_chat[:message_index])
                                          if entry["role"] == "user"), "")
                currencies = [str(product.get("currency") or "").strip().upper() for product in products]
                missing_price_currency = bool(re.search(r"\b(?:cheaper|cheapest|lower price|lowest price|price only|which costs less)\b",
                                                        previous_question, re.I)) and (
                    not all(re.fullmatch(r"[A-Z]{3}", currency) for currency in currencies)
                    or len(set(currencies)) != 1
                )
                if message["role"] == "assistant" and (missing_price_currency or any(
                    phrase in message["content"].casefold()
                    for phrase in ("not provided", "not returned", "missing or invalid", "different currencies", "details weren't provided", "couldn't find that detail", "no matching detail was returned", "units or format aren't clear")
                )):
                    for product_index, product in enumerate(products):
                        url = _retailer_url(product.get("url")) or _shopping_url(product.get("shopping_url"))
                        if url:
                            name = product.get("name") or f"Option {product_index + 1}"
                            st.link_button(f"Open {name} listing", url,
                                           key=f"chat-listing-{message_index}-{product_index}")


def render_comparison() -> None:
    header, home = st.columns([4, 1])
    with header:
        st.markdown('<div class="brand"><span class="brand-mark">✦</span><span>Deality AI</span></div>', unsafe_allow_html=True)
    with home:
        if st.button("← Home"):
            st.session_state.page = "home"
            st.rerun()
    st.title("Compare products")
    st.caption("Compare 2–5 products from different retailers. Search each option independently or enter listing details manually.")
    priority = st.selectbox("Recommendation priority", [*PRIORITIES, *SPEC_FOCUSES], key="shopping_priority")
    product_count = st.selectbox("Products to compare", [2, 3, 4, 5], key="product_count")
    result_area = st.empty()
    columns = st.columns(2)
    selected_by_slot = {}
    for number in range(1, product_count + 1):
        with columns[(number - 1) % 2]:
            st.markdown(f'<div class="eyebrow">OPTION {number}</div>', unsafe_allow_html=True)
            input_mode = st.radio(f"Input for option {number}", ["Search offers", "Manual details"],
                                  horizontal=True, key=f"input_mode_{number}", label_visibility="collapsed")
            if input_mode == "Search offers":
                with st.form(f"search_form_{number}"):
                    query = st.text_input(f"Search query {number}", key=f"query_{number}", placeholder="Enter a product or model")
                    submitted = st.form_submit_button(f"Search {number}", type="primary", width="stretch")
                if submitted:
                    _invalidate()
                    st.session_state[f"results_{number}"] = []
                    st.session_state.pop(f"selection_{number}", None)
                    clean_query = " ".join(query.split())
                    try:
                        if not clean_query:
                            raise ValueError("Enter a product search phrase.")
                        if clean_query not in st.session_state.shopping_cache:
                            with st.spinner("Searching Google Shopping…"):
                                st.session_state.shopping_cache[clean_query] = search_shopping_products(clean_query)
                        st.session_state[f"results_{number}"] = st.session_state.shopping_cache[clean_query]
                        if not st.session_state[f"results_{number}"]:
                            _notice("No results found. Try another search.")
                    except (SerpApiError, ValueError) as error:
                        _notice(str(error), "error")
                results = st.session_state[f"results_{number}"]
                if results:
                    index = st.selectbox(f"Product option {number}", range(len(results)), index=0,
                                         format_func=lambda i, items=results: _format_result(items[i]), key=f"selection_{number}")
                    selected_by_slot[number] = results[index]
            else:
                with st.form(f"manual_form_{number}"):
                    name = st.text_input(f"Product name {number}", key=f"manual_name_{number}")
                    category = st.text_input(f"Category or product type {number}", key=f"manual_category_{number}", placeholder="e.g. smartwatch")
                    seller = st.text_input(f"Retailer {number} (optional)", key=f"manual_seller_{number}")
                    product_url = st.text_input(f"Product page URL {number} (optional; never fetched)", key=f"manual_url_{number}")
                    price_text = st.text_input(f"Price {number} (optional; numbers only)", key=f"manual_price_{number}")
                    currency_text = st.text_input(f"Currency code {number} (optional, e.g. USD)", key=f"manual_currency_{number}")
                    rating_text = st.text_input(f"Rating {number} out of 5 (optional)", key=f"manual_rating_{number}")
                    reviews_text = st.text_input(f"Review count {number} (optional)", key=f"manual_reviews_{number}")
                    specs_text = st.text_area(f"Specifications {number} (one Name: value per line)", key=f"manual_specs_{number}", height=90)
                    manual_submitted = st.form_submit_button(f"Save option {number}", type="primary", width="stretch")
                if manual_submitted:
                    st.session_state[f"manual_{number}"] = None
                    _invalidate()
                    try:
                        if not name.strip() or not category.strip():
                            raise ValueError("Enter a product name and category to save a manual listing.")

                        def optional_number(raw, label, maximum=None, integer=False):
                            if not raw.strip():
                                return None
                            try:
                                value = float(raw.replace(",", "").strip())
                            except ValueError:
                                raise ValueError(f"{label} must be a number.") from None
                            if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum) or (integer and not value.is_integer()):
                                raise ValueError(f"{label} is outside the accepted range.")
                            return int(value) if integer else value

                        price = optional_number(price_text, "Price")
                        rating = optional_number(rating_text, "Rating", maximum=5)
                        reviews = optional_number(reviews_text, "Review count", integer=True)
                        currency = currency_text.strip().upper() or None
                        if currency and not re.fullmatch(r"[A-Z]{3}", currency):
                            raise ValueError("Currency must be a three-letter code, such as USD.")
                        specifications = {}
                        for line in specs_text.splitlines():
                            if not line.strip():
                                continue
                            if ":" not in line:
                                raise ValueError("Enter each specification as Name: value on its own line.")
                            key, value = line.split(":", 1)
                            if not key.strip() or not value.strip():
                                raise ValueError("Specification names and values cannot be blank.")
                            specifications[key.strip()] = value.strip()
                        price_display = f"{currency} {price:g}" if price is not None and currency else (f"{price:g} (currency unavailable)" if price is not None else "")
                        safe_url = _retailer_url(product_url)
                        digest = hashlib.sha256(f"{name.strip()}|{seller.strip()}|{safe_url}|{price_display}".encode("utf-8")).hexdigest()
                        st.session_state[f"manual_{number}"] = {
                            "id": f"manual-{digest}", "name": name.strip(), "category": category.strip(),
                            "seller": seller.strip(), "url": safe_url, "shopping_url": "", "image_url": "",
                            "price": price, "price_display": price_display, "currency": currency,
                            "rating": rating, "review_count": reviews, "description": "", "features": [],
                            "specifications": specifications, "source_note": "Manually entered by the user; not independently verified.",
                        }
                        st.session_state[f"results_{number}"] = []
                        st.session_state.pop(f"selection_{number}", None)
                        _invalidate()
                    except ValueError as error:
                        _notice(str(error), "error")
                manual_product = st.session_state.get(f"manual_{number}")
                if manual_product:
                    selected_by_slot[number] = manual_product
                    st.caption(f"Saved: {manual_product['name']} · User-entered details are unverified.")

    selected = [selected_by_slot[index] for index in range(1, product_count + 1) if index in selected_by_slot]

    # Invalidate the previous result whenever any selected option changes.
    pair = tuple(_identity(p) or repr(p) for p in selected) if selected else None
    if pair != st.session_state.current_pair:
        _invalidate()
        st.session_state.current_pair = pair
        st.session_state.pop("pair_confirmation", None)

    if len(selected) == product_count:
        duplicate = any(_same_result(first, second) for first, second in itertools.combinations(selected, 2))
        compatibility = comparison_compatibility(*selected)
        confirmed = False
        if duplicate:
            _notice("Select two different results; the same listing cannot be compared with itself.", "error")
        elif compatibility["status"] == "incompatible":
            _notice(compatibility["reason"], "error")
        elif compatibility["status"] == "uncertain":
            _notice(compatibility["reason"])
            confirmation_label = ("I confirm these two selected products serve the same purpose" if product_count == 2
                                  else "I confirm all selected products serve the same purpose")
            confirmed = st.checkbox(confirmation_label, key="pair_confirmation")
        if st.session_state.comparison and compatibility["status"] == "uncertain" and not confirmed:
            _invalidate()
        if st.button("Compare selected products →", type="primary"):
            _invalidate()
            if duplicate:
                _notice("Choose two distinct listings before comparing.", "error")
            elif compatibility["status"] == "incompatible":
                _notice("These product types cannot be compared.", "error")
            elif compatibility["status"] == "uncertain" and not confirmed:
                _notice("Confirm that this selected pair serves the same purpose before comparing.", "error")
            else:
                st.session_state.comparison = {"products": list(selected), "confirmed": confirmed, "priority": priority}

    comparison = st.session_state.comparison
    if not comparison:
        st.button("Clear searches, comparison and chat", on_click=_clear)
        _notice("Search for both products, select distinct results, and compare them.")
        render_assistant(priority)
        st.markdown('<div class="footer">This project is a prototype and is not financial advice.<br>Need help? Call <a href="tel:+16097514039">609-751-4039</a>.</div>', unsafe_allow_html=True)
        return
    products = comparison["products"]
    confirmed = comparison["confirmed"]
    comparison["priority"] = priority
    with result_area.container():
        st.markdown('<div class="eyebrow">YOUR COMPARISON</div>', unsafe_allow_html=True)
        summary = recommend_many(products, priority, confirmed=confirmed)
        winner, decision_label, explanation = _decision(products, summary, priority)
        title = products[winner].get("name") or f"Option {winner + 1}"
        st.markdown(f'<div class="recommendation"><div class="eyebrow">DEALITY AI · {escape(priority.upper())}</div><h3>{escape(decision_label)}: {escape(str(title))}</h3><p>{escape(explanation)}</p><p>{escape(summary["score_text"])}</p></div>', unsafe_allow_html=True)
        st.caption("Based on your priority, returned listing statistics, and customer feedback when available.")
        render_assistant(priority)
    rows = comparison_rows(products, confirmed=confirmed)
    if priority in SPEC_FOCUSES:
        focused = [row for row in rows if any(word in " ".join(row.values()).casefold() for word in SPEC_FOCUSES[priority])]
        if focused:
            st.dataframe(focused, hide_index=True, width="stretch")
    st.subheader("Side-by-side listings")
    product_columns = st.columns(2)
    for index, product in enumerate(products, 1):
        with product_columns[(index - 1) % 2], st.container(border=True):
            _render_product(product, index)
    st.subheader("Product details & specifications")
    st.caption("Details are shown as returned in search results; missing information is marked Not listed. Verify prices and availability with the retailer.")
    st.dataframe(rows, hide_index=True, width="stretch", height=520)
    st.caption("Search phrases go to SerpApi. Comparisons and chat stay in this Streamlit session. Listing details are not independently verified.")
    st.button("Clear searches, comparison and chat", on_click=_clear)
    st.markdown('<div class="footer">This project is a prototype and is not financial advice.<br>Need help? Call <a href="tel:+16097514039">609-751-4039</a>.</div>', unsafe_allow_html=True)


if st.session_state.page == "home":
    render_home()
else:
    render_comparison()
