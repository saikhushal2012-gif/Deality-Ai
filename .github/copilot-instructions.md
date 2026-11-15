# Deality AI Workspace Instructions

## Product and data flow
- Maintain the retailer-independent Python/Streamlit flow: search Google Shopping via SerpApi's `google_shopping` engine with one query per option, then compare 2–5 distinct selected results. Identical queries may use the session cache.
- Normalize only returned listing fields. Search results are not verified product-detail records; do not assume detailed specifications, features or direct merchant links exist. Display missing fields explicitly and never invent prices, currencies, ratings, reviews, features or specifications. Do not follow redirects to obtain merchant links.
- Block detected category mismatches; require explicit same-purpose confirmation for uncertain pairs. Category checks are heuristic, and the user is responsible for choosing genuinely comparable products.
- Keep rule-based recommendations transparent and AI prompts grounded in available, unverified listing evidence. Compare prices only when all currencies are known and match; review count is not a quality measure. Do not fabricate specification scores or unsupported winners. Do not claim ML training or evaluation without sourced data, a held-out test, and actual metrics.
- Use Deality AI branding consistently and preserve the blue/purple/cyan/pink gradient interface.

## Privacy and validation
- The app's comparison recommendation and chat use deterministic rules over selected listings; do not add hosted or local model providers without explicit product approval. Keep the limits clear: listing evidence is unverified and unsupported comparisons must not invent a winner.
- Load credentials from local `.env`; never print, expose or commit API keys. Search phrases go to SerpApi. Comparisons, cached results and chat remain in the Streamlit session without app-side persistence; provider data policies still apply.
- Validate changes with `python -m unittest discover -s tests` and `python -m compileall -q app.py comparewise tests`.
