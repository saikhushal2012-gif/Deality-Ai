# Deality AI Workspace Instructions

## Product and data flow
- Maintain the retailer-independent Python/Streamlit flow: search Google Shopping via SerpApi's `google_shopping` engine with one query per option, then compare 2–5 distinct selected results. Identical queries may use the session cache.
- Normalize only returned listing fields. Search results are not verified product-detail records; do not assume detailed specifications, features or direct merchant links exist. Display missing fields explicitly and never invent prices, currencies, ratings, reviews, features or specifications. Do not follow redirects to obtain merchant links.
- Block detected category mismatches; require explicit same-purpose confirmation for uncertain pairs. Category checks are heuristic, and the user is responsible for choosing genuinely comparable products.
- Keep the rule-based score transparent and hosted Gemini answers grounded in available, unverified listing evidence. Compare prices as truly cheaper only when currencies are known and match; when currency is unknown, describe only the lower/higher displayed amount. Review count is not a quality measure. Do not fabricate specification scores or unsupported product facts. Do not claim ML training or evaluation without sourced data, a held-out test, and actual metrics.
- Use Deality AI branding consistently and preserve the blue/purple/cyan/pink gradient interface.

## Privacy and validation
- Hosted Gemini chat is approved for contextual shopping answers; the numeric recommendation score remains deterministic. Send only bounded selected-listing fields, the question, and recent conversation; omit product/image URLs. Keep answers grounded in unverified listing evidence and never invent facts.
- Load credentials from local `.env` or Streamlit Cloud Secrets; never print, expose or commit API keys. Search phrases go to SerpApi. When configured, bounded questions, chat history and listing details go to Google AI. Comparisons, cached results and chat remain in the Streamlit session without app-side persistence; provider data policies apply.
- Validate changes with `python -m unittest discover -s tests` and `python -m compileall -q app.py comparewise tests`.
