# Deality AI

**Track:** AI + Finance  
**Team:** [Add every contributor’s name, email, and school]  
**Code:** [Add public GitHub repository or Colab link]

## The Problem

Online shoppers often compare similar products across different retailers, but prices, specifications, seller details, and ratings can be scattered or missing. Choosing the lowest listed price may not mean choosing the best fit or value. Deality AI helps shoppers review the available evidence before making a spending decision.

## What We Built

Deality AI is a Python/Streamlit prototype for comparing 2–5 offers across product categories—not only computers, but also watches, hoodies, appliances, headphones, cameras, and musical instruments. Users can search Google Shopping separately for each option or enter listing details manually. The app displays available listing details side by side, blocks detected product-purpose mismatches, and asks users to confirm uncertain comparisons.

The shopper immediately receives a transparent, deterministic recommendation based on the selected listings and priority. The inline Deality AI chat answers with comparison rules and supplied statistics; it makes no external model request. Shoppers can compare prices, ratings, review counts, or supplied specification evidence, while missing information remains marked unavailable.

## What We Added

Our additions include 2–5 product comparisons, manual entry when a search result is missing, heuristic compatibility checks, and cautious handling of incomplete listing data. Pasted retailer links are not fetched, currencies are not converted, and review count is not treated as product quality. The recommendation and chat are deterministic, use only current listing evidence, and show no winner when the evidence cannot support one. Deality Match remains a separate experimental ML classifier.

## How It Works

SerpApi’s Google Shopping Results API, using its `google_shopping` engine, retrieves offers. Results are unverified search listings, not product records confirmed by Deality AI. The app normalizes returned fields and user-entered details without filling in missing facts.

The app’s primary recommendation and inline assistant are deterministic comparisons of supplied listing evidence and selected priorities; no language model or AI provider is called at runtime. They do not independently verify retailer claims and return no winner if evidence is missing or tied. **Deality Match** is a separate experimental scikit-learn TF-IDF/logistic-regression preference classifier for same-category comparisons represented in its examples, including laptops and selected non-laptop categories. Non-laptop coverage is exceptionally sparse, so it is not a reliable general-purpose recommender.

## Results

The original laptop pilot contains 18 candidate listings in 6 scenarios; only 9 listings in 3 scenarios have explicit preference labels accepted for evaluation. That original classifier and cheapest-candidate baseline each matched **0/3** held-out favorites. Eleven added user-provided laptop scenarios contribute 27 candidates; fifteen scenarios from other categories contribute 35 candidates, stored separately. Combined grouped evaluation holds out whole scenarios: across 71 candidates and 29 scenarios, candidate-label accuracy was **0.521** and top-choice matches were **15/29**. Source subsets were **0/3** original laptop pilot, **6/11** added laptop scenarios, and **12/15** non-laptop scenarios. The non-laptop set is tiny, heterogeneous, user-provided, and unverified, so these results do not establish real-world performance. Separately, an offline experiment converted 20 user-provided synthetic scenarios (60 candidates across four categories) to the same labeled-row schema. With 16 whole training scenarios and 4 whole validation scenarios, candidate-label accuracy was **1.000 training / 0.667 validation**; scenario top-choice accuracy was **16/16 training / 1/4 validation**. The synthetic model is separate and is not used by the app. Automated tests use mocked search provider calls and do not establish live API or real-world recommendation quality.

## Limitations and Future Improvements

Our pilot listings, preference labels, and additional laptop and non-laptop scenarios are user-supplied and unverified; the separate 20-example set is synthetic. None is a public, license-checked real-world dataset, and non-laptop categories have only one or two scenarios each. Category checks are heuristic. Next, we would verify data sources and permissions, collect more labeled scenarios, test on unseen shoppers and products, compare against a cheapest-suitable baseline, document failures, and check live searches and mobile layouts. We will not claim real-world ML success from these data.

**This project is a prototype and is not financial advice.**

## Sources and Credits

- **Shopping search/API:** SerpApi, [Google Shopping Results API](https://serpapi.com/google-shopping-api). Pilot examples were supplied from retailer offers on [Walmart](https://www.walmart.com/), [Newegg](https://www.newegg.com/), [Best Buy](https://www.bestbuy.com/), and [Micro Center](https://www.microcenter.com/); listing details/prices are unverified and can change. No external public training dataset was used.
- **Recommendation assistant:** deterministic comparison rules; no hosted or local language model is called by the app.
- **Experimental model and software:** [scikit-learn](https://scikit-learn.org/), [Streamlit](https://streamlit.io/), [Requests](https://requests.readthedocs.io/), [python-dotenv](https://github.com/theskumar/python-dotenv), and [joblib](https://joblib.readthedocs.io/).
- **Development assistance:** [GitHub Copilot](https://github.com/features/copilot) assisted coding; [OpenAI ChatGPT](https://chatgpt.com/) assisted with prompt brainstorming/refinement. [Add any other AI tools, tutorials, assets, and sources used by the team.]
