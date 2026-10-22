# Bitcoin Narrative Risk

A Bitcoin workspace inside the main risk dashboard. It pairs **2,065 archived tweets, 258 Reddit posts and 730 daily price bars from 2021–2022** with FinBERT and VADER sentiment.

From the project root, run `streamlit run app.py`, then select **Bitcoin** under **Workspace**. Prices, sentiment, post details, themes and historical VaR share the same URL as the equity case study.

Saved results work offline. The Model lab downloads a pinned 111 MB quantized model on its first Analyze click, then runs on CPU. No API key or GPU is needed.

The two sentiment models disagree on about 65% of the posts. Next-day sentiment–return association is close to zero. The archives are selected samples with incomplete timestamp metadata, so the charts are exploratory.

Data: [Mendeley tweets](https://data.mendeley.com/datasets/8fbdhh72gs/5) and [Zenodo Reddit/prices](https://zenodo.org/records/12593440), both CC BY 4.0.

[Methods](METHODOLOGY.md) · [Findings](FINDINGS.md) · [Blogs and GitHub references](REFERENCES.md) · [Validation](VALIDATION.md)
