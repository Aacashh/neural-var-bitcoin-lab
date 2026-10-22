# Checked on 22 October 2026

Local Windows run using Python 3.13.5, PyTorch 2.6.0, Streamlit 1.45.1 and the installed scientific packages. Neural training used CPU and two PyTorch threads. A fresh isolated dependency installation was not repeated.

- Shared-dashboard suite: **33 passed in 37.92 seconds**. It covers risk calculations, causal model inputs, stress P&L reconciliation, all six equity sections, Bitcoin controls, switching workspaces in one session, direct Bitcoin links and equivalence of the feature-display cache to the original causal inputs. A fresh-process check verifies that browsing both workspaces imports neither PyTorch nor ONNX Runtime. After restoring the stress exposure-control assertion to its proper test, all **10 app tests passed in 16.05 seconds**.
- `python -m risklab audit`: **passed**. Reconstructed the curated returns from the raw provider ZIP, verified source/output/code checksums, reconciled actual portfolio losses and checked all 18,096 forecasts across 12 model/seed/quantile series and 1,508 common test sessions. Recomputed the benchmarks, train-only scaler and coverage metrics and replayed all three neural weight files.
- Opened the actual Streamlit server in the in-app browser, inspected the market overview, forecast and backtesting views, and adjusted plot legends, metric labels and rate/p-value formatting. Browser screenshots are retained in `screenshots/`.

The equity workspace reads frozen local data/results with no training or downloads in its page flow. Its input heatmap uses a checksummed feature cache built by the unmodified training feature function. The Bitcoin workspace also browses saved results; only its explicit Analyze button can download and load the optional CPU language model. No simulated returns were substituted.

These checks establish the behavior of this recorded build, not regulatory compliance, current market calibration, or general predictive superiority. See the case-study findings and methodology for the experiment's scope.
