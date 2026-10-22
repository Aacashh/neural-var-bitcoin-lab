# Bitcoin Narrative Risk

I added this tab to explore a different question: how does the language in Bitcoin posts compare with what the market actually did?

The demo uses **2,065 tweets, 258 Reddit posts and 730 daily Bitcoin price bars from 2021–2022**. These are already-collected public research archives. A quantized FinBERT model scores the text locally, with VADER as a simpler comparison. Scores are saved so the dashboard opens offline without loading the neural model.

## Open it

Run `python -m streamlit run app.py` from the project root, then choose **Bitcoin** under **Workspace**. Both case studies share the same app URL. Install `requirements.txt` for the demo or `requirements-dev.txt` for training, rebuilding scores and testing.

## What to try

- Switch between closing-price and OHLC charts, 2021 and 2022, and Twitter versus Reddit.
- Compare FinBERT with VADER. Open a disagreement and read the underlying language.
- Inspect sparse days and raise the minimum-post threshold. Missing sentiment stays missing.
- Look at volatility, drawdown, a historical loss threshold and descriptive lead–lag associations.
- Read the discovered TF-IDF/NMF themes and try one sentence in the local model lab.

## How the data is handled

Tweets come from selected crypto influencers in [Jahanbin, Zare Chahooki and Rahmanian's archive](https://data.mendeley.com/datasets/8fbdhh72gs/5). Reddit posts and BTC-USD OHLCV come from [George Vlahavas's archive](https://zenodo.org/records/12593440). Both archives are CC BY 4.0. We filter explicit whole-word `bitcoin`, `btc` or `bitcoins` mentions, restrict dates and remove exact normalized duplicate texts within each source. The original files, DOI metadata, transformation choices and checksums are retained.

The source files have 16,512 tweets and 907 Reddit posts before filtering. The keyword/date filter yields 2,363 posts and removes 40 duplicate texts, leaving 2,323. The Reddit description's count of 770 differs from its downloaded CSV; the app uses actual file counts. The Twitter archive has calendar dates only and starts its retained slice on 1 February 2021; do not infer exact publication times from the paper's broader description.

## Analysis choices

**Sentiment:** FinBERT returns positive, neutral and negative probabilities. The scalar score is `P(positive) - P(negative)`. The classes describe financial-language sentiment, not a prediction of Bitcoin's direction. VADER uses a lexicon and linguistic rules; it is a challenger, not a ground-truth label.

**Text:** Remove URLs and handles, retain hashtag words, negation and punctuation, and normalize whitespace. Use at most 128 tokens per post, including special tokens. Longer posts are truncated from the end; their flags and original token counts are visible. Input text uses the archive's supplied case and wording, which can already differ from the original post.

**Daily score:** The default is the mean post score each day. An alternative averages each available source's daily mean equally. If one source is missing on a date, only the source present contributes; the app exposes that coverage. The overall sentiment KPI averages eligible daily scores, giving each observed day equal weight. Minimum-post filtering hides low-count daily scores but leaves post counts and post-level evidence visible.

**Themes:** A 5-component NMF on TF-IDF features summarizes recurring terms. It is fit retrospectively on the full frozen corpus, for exploration only. Labels are the highest-weight words, not supervised topics or explanations of market moves.

**Market risk:** Daily returns are close-to-close. Realized volatility uses a trailing 30-day sample standard deviation, annualized with 365 for this daily crypto series. Drawdown uses the running maximum over the full price archive. Historical VaR is the 95th loss quantile from the prior 60 observed daily returns, using the higher order statistic. The target day never enters its own calibration window. The first 61 calendar days have no estimate. This is separate from the LSTM equity experiment.

**Lead–lag:** Positive lag pairs sentiment on t with the daily return at t+lag. Shift on the complete daily calendar before dropping missing scores, so a missing post day never turns into a one-day lead. The plot reports Spearman association and paired-day counts; it does not fit or evaluate a trading rule. Date filtering limits both members of a pair to the selected window.

**Limits:** Tweets are influencers; Reddit favors popular posts. Neither sample represents the whole platform. Dates lack complete timezone/publication metadata. Engagement is retrospective and is never used as a forecast weight. FinBERT was trained for financial text and has no validated accuracy on these crypto posts. Softmax confidence is not calibrated correctness. Sarcasm, mixed assets, slang and truncation matter. Correlation, disagreement and topic membership are not evidence of causality or profitable forecasting.

## Rebuild in a separate clean copy

```powershell
python -m pip install -r requirements-bitcoin.txt
python -m bitcoin fetch
python -m bitcoin model
python -m bitcoin score
python -m bitcoin audit
python -m bitcoin audit-model
```

Keep the included `data/`, `results/` and model manifests intact. Fetch and score are explicit setup operations and refuse to overwrite completed data/results. The model command safely restores missing files at the pinned revision and verifies their hashes against the saved manifest. A fresh data/score rebuild uses a separate copy without those outputs. Model weights are about 111 MB; the dashboard's saved scores are much smaller. The repository omits the large weights. The Model lab restores the pinned model and verifies its hashes on the first Analyze click; subsequent requests reuse the cached CPU runtime. Browsing saved results needs no model download. The CLI command `python -m bitcoin model` also restores the weights.

See [references and reuse details](REFERENCES.md) and [recorded validation](VALIDATION.md).
