# Deality AI — Submission Checklist

## Application and evidence
- [x] Retailer-independent Google Shopping search integration through SerpApi.
- [x] Compare 2–5 options; support manual product entry without fetching pasted URLs.
- [x] Preserve missing information honestly; show heuristic compatibility decisions and user confirmation when needed.
- [x] Show an immediate, deterministic recommendation and rule-based answers without external model setup.
- [x] Use blue/purple/cyan/pink Deality AI styling and include the required disclaimer in-app.
- [ ] Live-check at least two categories and multiple actual retailers with valid SerpApi access; capture real results and note date/region.
- [ ] Manually check desktop and mobile layouts in supported browsers.

## AI/ML claims
- [x] Distinguish deterministic comparison rules from Deality Match (an experimental, category-aware classifier with sparse non-laptop data).
- [ ] Select and cite a legally usable dataset with relevant, meaningful preference or utility labels.
- [ ] Document preprocessing, category boundaries, split strategy, leakage prevention, and reproducible training.
- [x] Report the Deality Match pilot honestly: 0/3 held-out scenario top-choice matches, equal to the cheapest-candidate baseline; do not claim improvement or reliability.
- [x] Report expanded Deality Match separately: 29 scenarios/71 candidates; grouped leave-one-scenario-out top-choice result 15/29 and candidate-label accuracy 0.521; separate original pilot (0/3), added laptops (6/11), and other categories (12/15), with sparse unverified data clearly qualified.
- [x] Keep the separate synthetic experiment distinct: 1.000/0.667 candidate-label training/validation accuracy and 16/16 vs 1/4 scenario top-choice accuracy; do not present synthetic validation as real-world accuracy.
- [x] Do not submit ML improvement claims without stronger data and evaluation.

## Required team/submission information
- [ ] Confirm official hackathon rules, rubric, deadlines, and exact submission format.
- [ ] Finance track selection: AI + Finance (verify wording against rules).
- [ ] Add all team members' full names, approved emails, and schools.
- [ ] Publish and review a public GitHub repository; add its URL to the one-pager.
- [ ] Verify all dataset, model, library, asset, tutorial, and AI-tool credits against actual use and license requirements.
- [ ] Record an optional demonstration video of no more than two minutes using the real application; include truthful narration about incomplete ML work.
- [x] Exact disclaimer included: "This project is a prototype and is not financial advice."

## Submission files
- README: project purpose, setup, APIs, limitations, credits, and truthful ML status.
- One-page project summary: `HACKATHON_ONE_PAGER.md`.
- This completion checklist: `HACKATHON_SUBMISSION_CHECKLIST.md`.
