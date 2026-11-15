# Deality AI

Deality AI is a hackathon prototype for comparing product offers from multiple retailers and helping shoppers reason about overall value. It is a Streamlit application with a blue, purple, cyan, and subtle pink gradient interface. The current live search source is SerpApi's Google Shopping Results API; Amazon is only one possible retailer among others.

> This project is a prototype and is not financial advice.

## Run locally

1. Install Python 3.10 or later, then install dependencies with `python -m pip install -r requirements.txt`.
2. Copy `.env.example` to `.env`. Add your own `SERPAPI_API_KEY` for live Shopping search. This key is separate from the AI key.
3. Optional hosted assistant: set `GEMINI_API_KEY` and, if desired, `GEMINI_MODEL` in `.env`. The default model is `gemini-3.8-flash`; model access, free-tier eligibility, quotas, pricing, and data terms can change. Check [Google AI Studio pricing](https://aistudio.google.com/docs/pricing) before use. Gemini requests require explicit in-app consent and only happen when a user asks a question or requests an opinion.
4. Optional local assistant: install [Ollama](https://ollama.com/download), pull `qwen2.5:3b`, and set `OLLAMA_MODEL` if using a different locally installed model. No hosted AI key is required for this option. Ensure the Ollama server is actually configured for local inference.
5. Run `python -m streamlit run app.py` and open `http://localhost:8501`. Restart after changing `.env`.

Never commit `.env`, print a key, or share credentials in chat. `.gitignore` excludes the local environment file. Do not deploy this prototype with API keys exposed to browsers or public source control.

## Product comparison

Choose 2–5 products. For each option, either search Google Shopping independently or enter listing details manually. Manual product URLs are displayed only when they pass conservative URL checks; Deality AI never fetches a pasted product URL or follows redirects. Search uses SerpApi's `google_shopping` engine and identical search phrases can reuse results cached only in the current Streamlit session.

The app blocks detected product-purpose mismatches and duplicate listings. Uncertain category matches require explicit user confirmation. These category checks are heuristics, so users remain responsible for choosing genuinely comparable products.

Shopping results are unverified offers, not authoritative product-detail records. The app uses only fields returned by the provider or entered by the user. Missing prices, currencies, ratings, reviews, features, specifications, retailer links, and images remain unavailable rather than being fabricated. Verify offer details, availability, shipping, and returns with the retailer. Ratings may describe a product rather than its seller.

## Recommendation method and ML status

The app's primary recommendation is generated on request by the selected language model: local Ollama or, when configured and consented to, Google Gemini. The response is grounded in selected listing fields, the chosen priority, and recent chat, but it can be wrong and is not independently verified. **Deality Match** is a separate experimental scikit-learn classifier for same-category comparisons represented in its labeled examples (laptops, smartphones, smartwatches, apparel, headphones, shoes, speakers, e-readers, power banks, backpacks, rain jackets, and air fryers). Non-laptop coverage is exceptionally sparse—one or two scenarios per category—so recommendations are highly experimental and not reliable. A transparent, comparison-local deterministic score is shown separately as a **rule-based baseline**. It uses supplied prices with known matching currencies, ratings, or review counts when the selected priority calls for them; review count is not treated as product quality, and specification priorities do not assign unsupported spec scores.

The original laptop pilot dataset is in `data/labeling_pilot.csv`; its listings and labels are user-provided and unverified. Eleven additional user-provided laptop scenarios are in `data/deality_match_scenarios.json`; 15 cross-category scenarios (35 candidate listings) are in `data/deality_match_other_scenarios.json`. The app fits a TF-IDF/logistic-regression classifier in memory when requested. Run `python scripts/train_deality_match.py` to evaluate with whole-scenario holdouts: across 29 scenarios/71 candidates, candidate-label accuracy was 0.521 and top-choice matches were 15/29. The original pilot remains 0/3; added laptop scenarios 6/11; added non-laptop scenarios 12/15. The non-laptop result comes from a tiny, heterogeneous dataset, and none of these numbers is a reliable performance estimate. Details and labels are user-provided, unverified; do not claim ML improvement or real-world accuracy. Fit-on-all resubstitution accuracy is omitted because it is not an estimate of generalization.

A **separate offline synthetic experiment** is stored in `data/synthetic_preference_examples.json`. Run `python scripts/train_synthetic_model.py` to convert its 20 shopper scenarios (60 candidate rows across laptops, watches, hoodies, and musical instruments) into the existing labeled-candidate schema and train a separate local artifact. It uses 16 whole scenarios (48 candidates) for training and holds out four entire scenarios (12 candidates)—one per category—for validation; no scenario's candidates are split between the sets. Candidate-label accuracy was **1.000 training** and **0.667 validation**. Scenario top-choice accuracy was **16/16 training** and **1/4 validation**; the top-choice result is the more direct measure of selecting each scenario's preferred product. This synthetic artifact is not used by the app or Deality Match, and these examples do not establish real-world recommendation accuracy. The target preference is excluded from features, as is the accompanying rationale. Features use the shopper answer/profile, question, budget and within/over-budget signal, priorities, product category/name, and supplied product attributes. No separate real-world test set was supplied; validation scenarios are held out from fitting.

## AI assistant and privacy

The interface offers local Ollama or Google Gemini when a Gemini key is configured. These are generative assistants, distinct from Deality Match and the rule-based baseline; their answers may be wrong. The app grounds prompts in available listing fields and bounded chat history, but does not independently verify retailer claims. Gemini sends selected listing details, the question, priority, and recent chat to Google after in-app consent. Free-tier usage is not necessarily unlimited or private; review Google's current terms and usage page. Ollama requests go to loopback, but the app cannot verify whether an independently configured Ollama server forwards data. Avoid entering sensitive information.

Search phrases are sent to SerpApi, subject to that provider's data policies and quota. Product selections, cache, comparison state, and chat are kept in the active Streamlit session and are not written to a Deality database. **Clear searches, comparison and chat** removes the session's current comparison data.

## Tests and verification

Run `python -m compileall -q app.py comparewise tests scripts` and `python -m unittest discover -s tests`. Tests mock provider HTTP calls; they do not prove live SerpApi/Gemini availability, accurate current offers, or recommendation quality on real-world user outcomes. Live tests require credentials and can consume quota.

## Credits and submission status

- Product listing source: SerpApi Google Shopping Results API. Search result content and provider terms belong to their respective sources. The pilot and scenario examples in `data/labeling_pilot.csv`, `data/deality_match_scenarios.json`, and `data/deality_match_other_scenarios.json` were supplied by the project user in this workspace and have not been independently verified; confirm source/usage permissions and cite underlying listing sources before public submission. `data/synthetic_preference_examples.json` contains user-provided synthetic examples.
- AI APIs/models: Google Gemini Developer API (optional, configured model noted above); Ollama and `qwen2.5:3b` (optional local inference, model distributed under its own license). Verify each current model's license and attribution requirements for the submission.
- Libraries: Streamlit, Requests, python-dotenv, scikit-learn, and joblib; see `requirements.txt` for package/version ranges and each project's license.
- Development assistance: GitHub Copilot and OpenAI ChatGPT were used for code assistance and prompt brainstorming/refinement. Team members should add any other tools, tutorials, assets, and sources they use and follow the hackathon's attribution rules.
- Track: AI + Finance (team to verify the exact track name in the current rules).
- Public repository, team contact details, dataset credits, and measured ML results must be added by the team before submission; none are invented here.
