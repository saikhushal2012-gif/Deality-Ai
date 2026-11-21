"""Verified project facts and concise offline answers for judge preparation."""

from __future__ import annotations

import re


PROJECT_BRIEF = """
Deality AI is a Python and Streamlit prototype for comparing two to five shopping listings.
Users search Google Shopping through SerpApi's google_shopping engine, one query per option,
or enter listing details manually. It displays returned listing fields and blocks detected
purpose mismatches; uncertain pairs require user confirmation. Search listings and user-entered
facts are not independently verified. Missing facts are not invented, currencies are not
converted, and review counts are not treated as product quality.

The live recommendation score is deterministic and uses the selected priority and available
listing evidence. Direct price and weight questions use local comparison rules. When
GEMINI_API_KEY is configured, open-ended shopping chat and judge-preparation answers may use
Google Gemini. The question and bounded recent conversation are sent for judge Q&A; shopping
chat also sends bounded selected-listing fields, but not product or image URLs. When Gemini is
not configured or fails, the app uses local rules or the offline judge-answer reference.

Deality Match is a separate offline scikit-learn experiment using TF-IDF and logistic
regression; it is not used by the live app. In its grouped evaluation across 29 scenarios and
71 candidates, candidate-label accuracy was 0.521 and the preferred top choice matched in
15/29 scenarios. Results varied by subset. A separate synthetic experiment had 1.000 training
and 0.667 validation candidate-label accuracy, with 16/16 training and 1/4 validation
scenario top-choice matches. These small or synthetic results do not establish real-world
recommendation quality.

Live search phrases go to SerpApi. Product selections, cache, comparison state, and chats stay
in the active Streamlit session; the app does not write them to a Deality database. Provider
data policies apply to information sent to SerpApi or Google AI. Automated tests use mocked
provider responses and do not establish live service availability, offer accuracy, or actual
consumer savings. The project has not measured savings or shown that recommendations improve
real purchasing outcomes. Currency conversion, independent listing verification, and a
validated general-purpose ML recommender are not implemented. Affiliate revenue and premium
features are only possible future business models, not current features.
""".strip()


_ANSWERS = (
    (re.compile(r"\b(?:pitch|introduce|what is deality|tell me about (?:the )?project)\b", re.I),
     "Deality AI is a Streamlit app that brings two to five shopping offers into one comparison. "
     "It helps shoppers understand price and available listing evidence instead of focusing only "
     "on the lowest advertised price. Its recommendations are deterministic, and it keeps missing "
     "information visible rather than inventing facts."),
    (re.compile(r"\b(?:finance|financial|save(?: money)?|savings|spending|personal finance)\b", re.I),
     "Everyday purchasing is part of personal finance. Deality AI is intended to make trade-offs "
     "between listed prices and available product evidence easier to understand before spending. "
     "We have not measured actual savings, so we describe that as a potential benefit, not a proven result."),
    (re.compile(r"\b(?:google shopping|why not (?:just )?google|shopping search)\b", re.I),
     "Google Shopping helps people discover offers. Deality AI focuses on the next step: users "
     "select multiple offers, compare returned information side by side, set a priority, and see "
     "whether the available evidence supports a recommendation."),
    (re.compile(r"\b(?:who (?:would|will) benefit|target users|who is it for)\b", re.I),
     "It is intended for people comparing offers across retailers, especially for higher-cost "
     "purchases such as laptops, electronics, and appliances. We have not yet validated outcomes "
     "with a broad user study."),
    (re.compile(r"\b(?:where is the ai|does (?:the )?chat|chat assistant|chatgpt|gemini|language model|hosted model)\b", re.I),
     "The live recommendation score is deterministic. Direct price and weight answers are also "
     "rule-based. If GEMINI_API_KEY is configured, open-ended shopping chat and this judge Q&A "
     "mode can use Gemini; if it is unavailable, the app falls back to local rules or project notes."),
    (re.compile(r"\b(?:rule[- ]based|why (?:did|do) you use rules|machine[- ]learning model|ml model|trained)\b", re.I),
     "The live recommendation is rule-based so its behavior is transparent and does not depend "
     "on a model trained from limited data. We ran separate TF-IDF/logistic-regression experiments, "
     "but they are offline research and do not power the app."),
    (re.compile(r"\b(?:accuracy|evaluation|metrics|held[- ]out|classifier|machine learning|\bml\b)\b", re.I),
     "The separate Deality Match experiment used whole-scenario holdouts across 29 scenarios and "
     "71 candidates: candidate-label accuracy was 0.521 and top-choice matches were 15/29. A "
     "separate synthetic experiment scored 1.000 on training and 0.667 on validation for "
     "candidate labels, with top-choice results of 16/16 training and 1/4 validation. These small "
     "or synthetic results do not establish real-world performance. The classifier is not used "
     "in the live app."),
    (re.compile(r"\b(?:where (?:does|do) (?:the )?(?:product )?information come from|data source|serpapi|google shopping results)\b", re.I),
     "Live offers come from SerpApi's Google Shopping Results API. Users may also enter listing "
     "details manually. The app displays returned or user-entered information; it does not "
     "independently verify retailer listings."),
    (re.compile(r"\b(?:accurate|verify|verified|trust the product|correct is the information)\b", re.I),
     "We do not independently verify every listing. The app uses information returned by SerpApi "
     "or entered by the user, and missing details remain unavailable. Shoppers should verify "
     "price, availability, shipping, returns, and specifications with the retailer."),
    (re.compile(r"\b(?:different currenc|currency conversion|convert currencies)\b", re.I),
     "The app does not convert currencies and does not make a true cheaper/more-expensive price "
     "comparison when currencies differ or are unknown. It can describe displayed amounts without "
     "claiming they are directly comparable."),
    (re.compile(r"\b(?:rating|review count|more reviews|higher rating)\b", re.I),
     "The app shows supplied ratings and review counts as separate listing fields. It does not "
     "treat more reviews as proof of quality, and missing ratings remain unavailable."),
    (re.compile(r"\b(?:not comparable|category|same purpose|mismatch|different products)\b", re.I),
     "Heuristic compatibility checks block detected purpose mismatches and ask for confirmation "
     "when a comparison is uncertain. They are safeguards, not proof that products are equivalent; "
     "the user remains responsible for choosing comparable items."),
    (re.compile(r"\b(?:technolog|tech stack|built with|libraries|framework)\b", re.I),
     "The app uses Python and Streamlit, Requests for HTTP calls, python-dotenv for local "
     "configuration, and SerpApi for Google Shopping search. scikit-learn and joblib support "
     "separate offline experiments."),
    (re.compile(r"\b(?:how (?:did|do) you test|testing|test suite|mocked)\b", re.I),
     "Automated tests cover listing normalization, comparison rules, missing data, currencies, "
     "compatibility, manual listings, chat behavior, and Streamlit flows. Provider calls are "
     "mocked, so passing tests do not prove live API availability, current offer accuracy, or savings."),
    (re.compile(r"\b(?:api (?:fails|failure|stops)|quota|provider unavailable|search unavailable)\b", re.I),
     "Live search depends on SerpApi and can fail or reach quota limits; users can enter listing "
     "details manually. If Gemini is unavailable, open-ended answers fall back to local rules or "
     "the judge Q&A reference."),
    (re.compile(r"\b(?:privacy|user information|store conversations|data stored|database)\b", re.I),
     "Comparison state and chat stay in the active Streamlit session and are not written to a "
     "Deality database. Search phrases go to SerpApi. If Gemini is configured, judge questions "
     "and bounded conversation go to Google AI; shopping chat also sends bounded listing fields, "
     "excluding product and image URLs. Provider data policies apply."),
    (re.compile(r"\b(?:deployed|deployment|streamlit cloud|live app)\b", re.I),
     "Project notes record an initial Streamlit Community Cloud deployment, but the current "
     "deployment status should be checked before presenting. The source app requires its provider "
     "secrets to be configured in the hosting environment."),
    (re.compile(r"\b(?:limitations|biggest challenges|future work|improve next|next steps)\b", re.I),
     "Current limitations include incomplete and unverified listing data, no currency conversion, "
     "heuristic category matching, and reliance on SerpApi. Savings are not measured, and the "
     "experimental classifier is not validated for general use. Next steps include broader user "
     "testing, stronger product matching, and evaluation on permitted, representative data."),
    (re.compile(r"\b(?:make money|business model|revenue|affiliate|premium)\b", re.I),
     "Possible future models include transparent affiliate partnerships, premium comparison "
     "features, or retailer partnerships. None is implemented. Any sponsored relationship would "
     "need clear disclosure to protect user trust."),
    (re.compile(r"\b(?:innovative|what makes .*different|unique|why is this useful)\b", re.I),
     "The project focuses on the decision after offer discovery: selecting listings, comparing "
     "available evidence, setting a priority, applying compatibility safeguards, and explaining "
     "when the evidence does not support a clear winner."),
)


def answer_project_question(question: str) -> str:
    """Return a concise, verified local answer when Gemini is unavailable."""
    prompt = " ".join(str(question or "").split())
    if not prompt:
        return "Ask a question about Deality AI, its design, technology, results, or limitations."
    for pattern, answer in _ANSWERS:
        if pattern.search(prompt):
            return answer
    return (
        "I don't have a verified project-specific answer for that yet. The safe summary is: "
        "Deality AI compares unverified shopping listings with deterministic rules; Gemini is "
        "optional for open-ended chat, and the ML experiments are offline rather than part of "
        "the live recommendation."
    )