# Checked on 22 October 2026

## Data and inference

- Public source files are frozen with SHA-256 hashes and their DOI/license metadata. The tweet archive was additionally checked against the publisher's SHA-256. Model files were checked against official repository blob identities at the pinned revision.
- The final data has **2,323 posts** (2,065 Twitter; 258 Reddit) and **730 complete daily OHLCV bars**. No invented text, synthetic price history, forward-filled sentiment or live scraping was substituted.
- FinBERT ran locally with ONNX Runtime 1.23.2, CPUExecutionProvider, two intra-operation threads, batches of 16 and a 128-token limit. VADER 3.3.2 ran on the same cleaned text. No network sentiment API was used.
- A complete second inference build replayed **all 2,323 texts** after improving restoration of missing model files. Class probabilities and scalar scores matched the first build within `1e-7`; labels and discovered themes matched. Both manifests are preserved, and the final numerical audit passed. See `results/replay_receipt.json`.
- `python -m bitcoin audit` verifies current code/input/output identities, raw-source checksums, unique retained text IDs, probability sums, label/score reconciliation and independently recomputed market metrics. Full model inference is a separate, explicitly requested operation; the replay receipt records the complete second pass.

## Application checks

- Shared-entry-point VaR + Bitcoin suite: **33 tests passed in 37.92 seconds**. It includes same-session workspace switching, a Bitcoin query link, feature-cache equivalence and a fresh-process check that saved-result browsing loads neither the training nor the inference runtime.
- After the final Bitcoin chart/label and local-inference interaction changes: **12 Bitcoin tests passed in 18.70 seconds**. These cover missing sentiment versus real neutrality, source weighting, minimum-count scope, calendar-preserving lead/lag alignment, prior-day VaR calibration, text cleaning, data counts, source filters including an empty selection, date/model/aggregation controls, candles and a no-post date window.
- The earlier equity VaR numerical audit also passed again: 18,096 forecasts, all three saved LSTM models, 1,508 common held-out sessions and benchmark/scaler/metric replay.
- Inspected the actual Streamlit app in the in-app browser: dated KPIs, aligned price/sentiment/count panels, model-review controls and local sentence analysis. Screenshots are saved in `screenshots/`.

The environment is Windows / Python 3.13.5. The local venv reuses existing scientific packages and adds the pinned sentiment runtime. A completely isolated fresh installation was not independently tested. Some sandboxed pytest processes emitted an ignored `pyreadline3` console-cleanup warning after successful tests; the final tests exited successfully and browser operation was unaffected.

These checks verify this case-study build. They do not establish sentiment-label accuracy, model calibration, causal relationships, predictive trading performance, or production readiness.
