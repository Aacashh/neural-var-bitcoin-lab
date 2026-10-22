# What the frozen replay shows

This is a descriptive case study of 2,323 Bitcoin-keyword posts and 730 daily price bars in 2021–2022. Default settings are FinBERT, both sources, daily post mean and at least one retained post per scored day.

- There are **646 days with sentiment**, of which **185 have both sources**. Twitter supplies 2,065 posts across 617 dates; Reddit supplies 258 posts across 214. The data do not measure platform-wide attention.
- FinBERT assigns **310 positive, 1,838 neutral and 175 negative** labels. Its language-neutral class dominates this sample. Without human labels, that cannot be called either accurate or inaccurate.
- FinBERT and VADER disagree on **1,516 posts, or 65.3%**. This is a useful manual-review queue, not an error rate or proof that one model is better.
- **166 posts (7.1%)** exceed the 128-token input limit, and **801 (34.5%)** have maximum softmax probability below .65. Truncation is visible at post level; confidence has not been calibrated against crypto-social ground truth.
- Spearman association between the daily FinBERT score and the **same-day return is 0.142** across 646 paired dates. With the **next calendar day's return it is −0.016** across 645 pairs. These comparisons are descriptive and unadjusted; they do not establish prediction or causation. Publication times are also incomplete.
- Bitcoin's archived closing price moves from **$29,374 to $16,547**, a **43.7% decline** between the first and last selected closes. The maximum close-based drawdown from an in-archive prior peak is **76.6%**. This differs from the first-to-last return.

The first full CPU scoring pass took **236.8 seconds**, plus **2.3 seconds** for model initialization. The dashboard uses cached scores and does not repeat this computation on widget changes. A second complete replay verifies the final build after improving missing-model-file restoration; its receipt and current timing are recorded with the results.

The next useful step is a human-labeled review set, stratified by source, sentiment, confidence and truncation. A predictive study would additionally need reliable availability times, a preregistered chronological holdout, price-only baselines and costs. Adding a larger model before addressing these issues would not settle the core question.

See [references](REFERENCES.md) for the archives, model and dashboard sources.
