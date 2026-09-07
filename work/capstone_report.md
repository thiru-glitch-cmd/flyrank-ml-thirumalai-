# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** FlyRank ML Intern
- **Lane:** Refresh / Content Opportunity Scoring (Lane 2)
- **Repo:** https://github.com/flyrank-bih/flyrank-ml-internship-starter
- **Date:** 2026-07-15

## 0. Abstract

This capstone project addresses the problem of organic search content decay by building a repeatable machine learning prioritization model. Leveraging a public-safe dataset of 30,000 pages across 32 clients from the FlyRank Search Intelligence platform, we build features from trailing 90-day search performance and content attributes to predict whether a page's impressions will decline in a future window. We evaluate Logistic Regression, Decision Tree, and Random Forest models on a client-holdout split, finding that the Random Forest model achieves a Precision@50 of **0.680** (a **2.8x lift** over the rule-based baseline Precision@50 of **0.240**). The final output is an explainable ranked refresh queue with action tags and reason codes that helps content editors review and optimize high-value content efficiently.

## 1. Problem framing

Organic search content naturally decays over time due to shifts in search intent, seasonal trends, and competitor updates. Audit teams face the problem of prioritization: out of thousands of published articles, which ones should a human editor review and refresh first? 

- **Unit of Analysis:** A unique content item (page) aggregated over a trailing 90-day window.
- **Output:** A ranked prioritization list with predicted decline probabilities, action tags, and readable reason codes.
- **Action:** A human editor audits the page to rewrite stale sections, adjust meta descriptions, optimize keywords, or merge thin articles.
- **Cost of Wrong Call:**
  - *False Positive:* Wasting scarce editorial budget and hours rewriting content that does not need a refresh.
  - *False Negative:* Missing a decaying high-value article, leading to permanent loss of organic traffic and revenue.
- **Why ML Helps:** Search performance factors (impressions, positions, CTR, content age, word count, user engagement) interact in non-linear ways. Simple if-then rules are too rigid to capture these multi-variable relationships across diverse client industries, whereas machine learning models can find subtle patterns and adjust probabilities dynamically.

## 2. Data safety

The dataset is a public-safe slice containing **30,000 rows** across **32 distinct clients**.
- **Deliberately Excluded Columns:**
  - `trend_direction` and `trend_pct`: These columns are directly used to calculate the label. Keeping them as features would lead to 100% target leakage.
  - `content_id` and `client_id`: Used purely for indexing, joining, and grouping splits.
  - `provider_used` and `model_used`: LLM context tags that are not related to organic search performance.
  - Raw URLs, titles, client names, and search queries were hashed or pseudonymized before release to protect privacy.
- **Leakage Audits:** We verified that no feature was computed after the decision point. The train/test split is grouped by client (`client_id`) to ensure no client overlap between training and testing.

## 3. Baseline

The baseline is a rule-based priority score computed as:
$$\text{Baseline Score} = 0.40 \times \text{Visibility} + 0.30 \times \text{Freshness Risk} + 0.25 \times \text{Position Opportunity} + 0.05 \times \text{Depth Gap}$$

This is a fair comparison because it represents the actual heuristic rules typically implemented in production SEO dashboards to prioritize pages. On the same client-holdout split:
- **Baseline Precision@50:** **0.240**
- **Baseline ROC AUC:** **0.627**
- **Baseline Average Precision (AP):** **0.468**

## 4. Model / analysis

We evaluated three model architectures using scikit-learn:
1. **Logistic Regression** (with standard scaling and balanced class weights)
2. **Decision Tree Classifier** (max depth = 5, min samples leaf = 50)
3. **Random Forest Classifier** (200 estimators, max depth = 10, min samples leaf = 25)

The target is `is_declining_label`, defined as `trend_direction == 'down'` (which has a base rate of **54.2%** in the dataset).

## 5. Evaluation

To guarantee the models generalize to unseen website inventories, we use a **client-holdout split** (80% of clients used for training, 20% held out for testing).

| Model | ROC AUC | Average Precision | Precision@50 | Recall | F1 |
|---|---|---|---|---|---|
| **Random Forest** | **0.747** | **0.610** | **0.680** | **0.741** | **0.638** |
| Decision Tree | 0.742 | 0.575 | 0.620 | 0.716 | 0.634 |
| Logistic Regression | 0.700 | 0.522 | 0.400 | 0.567 | 0.566 |
| Baseline Rules | 0.627 | 0.468 | 0.240 | 0.189 | 0.274 |

- **Error Analysis:** The baseline rules suffer from very low recall (18.9%), failing to catch a large portion of decaying pages because they rely on rigid thresholds. The Random Forest model yields a **183% lift in Precision@50** over the baseline, meaning a reviewer's time is utilized almost 3 times more effectively.

## 6. Interpretation

The Random Forest model identified the following top feature importances:
1. `days_with_impressions` (16.1%): Indicates consistency of search exposure.
2. `log_impressions_90d` (12.8%): Measures scale of visibility.
3. `avg_position` (10.8%): Measures search rank; lower (better) ranks are highly predictive of stability.
4. `content_age_days` (9.5%): Captures natural temporal decay.
5. `word_count` / `char_count` (approx. 4.0% each): Measures content depth.

This indicates that visibility consistency (`days_with_impressions`) and search volume scale are the strongest predictors of content performance stability.

## 7. Recommendation

Editors should use the ranked priority queue to plan their weekly refresh schedules:
1. Filter the queue for `Action = refresh_and_review_ctr` or `expand_and_refresh`.
2. Inspect the top 50 pages where the model probability is highest.
3. Use the provided reason codes (e.g. `declining_with_demand`, `ctr_review_candidate`) to decide the action:
   - For `ctr_review_candidate`: Focus on meta title/description rewrites and snippet optimization.
   - For `stale_visible_page`: Focus on updating factual information and adding new sections.
- **Limits:** The model should not be used as an auto-publisher. Human review is required to verify if the query intent has changed or if competitor pages have structurally changed.

## 8. Reproducibility

To reproduce the analysis from a fresh clone:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install reportlab
   ```
2. Run the complete pipeline script:
   ```bash
   python scripts/run_all.py
   ```
- Random seed used: `42` for all data splits and model training.
- Outputs are saved in `outputs/` and processed tables in `data/processed/`.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset linking to [https://flyrank.ai](https://flyrank.ai).
