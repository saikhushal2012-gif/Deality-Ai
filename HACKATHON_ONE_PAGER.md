# Deality AI — Hackathon Project Summary

**Track:** AI + Finance (verify the exact track name in the current rules)  
**Team:** Add each member's approved full name, email, and school before submission.  
**Repository:** Add the public repository URL after the team publishes and reviews it.

## Problem

Shopping across retailers is fragmented. A low advertised price does not necessarily mean the best overall value, while offer pages may omit specifications, availability, shipping, or other decision-critical details.

## Solution

Deality AI is a Streamlit prototype for comparing 2–5 product offers. Shoppers can search Google Shopping through SerpApi or enter listing details manually. The app displays only information returned by the provider or entered by the user, checks product-purpose compatibility, and places prices, ratings, seller information, and available specifications side by side. It makes no claim that search listings are independently verified.

## AI approach and current status

The app immediately provides a deterministic recommendation from the selected priority and returned listing evidence, and its chat uses the same rules without contacting an external AI service. It reports no winner when the evidence is insufficient or tied. Deality Match is a separate experimental scikit-learn preference classifier for supported same-category products, trained from user-provided examples; non-laptop category coverage is very sparse.

The original laptop pilot has 9 eligible candidate rows across 3 scenarios; its classifier and cheapest-candidate baseline each matched 0/3 held-out favorites. Deality Match now uses 71 candidate rows across 29 scenarios: the original pilot, eleven additional laptop scenarios, and fifteen scenarios across other categories. Leave-one-scenario-out across all examples gave 0.521 candidate-label accuracy and 15/29 top-choice matches; results by source were 0/3 original laptop pilot, 6/11 added laptop scenarios, and 12/15 non-laptop scenarios. The non-laptop sample is highly sparse and heterogeneous; these unverified examples do not establish real-world performance. Separately, an offline synthetic experiment had candidate-label accuracy of 1.000 training / 0.667 validation and scenario top-choice accuracy of 16/16 training / 1/4 validation. We will not claim improved recommendations.

## Data and testing

Live search phrases are sent to SerpApi's Google Shopping Results API. Search results are provider listings, not verified detail pages. The original pilot and added laptop scenarios were supplied by the project user and are unverified; a separate 20-scenario synthetic dataset is also user-provided. No external public training dataset is used. Automated tests use mocked provider responses and cannot establish current offer accuracy or recommendation quality on real user outcomes.

## Results and limitations

The experimental classifier has only the small pilot result above; the deterministic rules are transparent and are not a learned recommendation model. Product comparisons are limited to returned/manual fields and heuristic category checks. Pasted URLs are not fetched. SerpApi credentials and quota are required for live discovery. Live cross-category and retailer verification remains to be demonstrated before submission.

## Future work

Collect permitted, verified preference labels; evaluate the generated and learned recommendations on unseen scenarios against a cheapest-suitable baseline; test live searches with permission; and validate responsive behavior on target devices.

> This project is a prototype and is not financial advice.

## Credits

- Listing discovery: SerpApi Google Shopping Results API; provider data is subject to its own terms.
- Recommendation assistant: deterministic comparison rules; no language model is used at runtime.
- Software libraries: Streamlit, Requests, python-dotenv, scikit-learn, and joblib; check package licenses in the installed distributions.
- Development tools: GitHub Copilot assisted coding; OpenAI ChatGPT assisted prompt brainstorming/refinement. Add any other AI tools, assets, tutorials, and sources used by the team.
- Experimental pilot data: user-provided, unverified listing and preference examples in `data/labeling_pilot.csv`; no external public training dataset.
