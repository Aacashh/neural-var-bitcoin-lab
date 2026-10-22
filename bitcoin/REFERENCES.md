# References and what I used

## Data

1. **Jahanbin, K.; Zare Chahooki, M. A.; Rahmanian, F. (2023).** *Database of influencers' tweets in cryptocurrency (2021–2023)*, Mendeley Data, version 5. [Dataset](https://data.mendeley.com/datasets/8fbdhh72gs/5), [DOI](https://doi.org/10.17632/8fbdhh72gs.5), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used the raw `full_text` and date fields from the 16,512-row CSV. Existing sentiment outputs were ignored. Changes: Bitcoin-keyword and date filtering, URL/handle cleaning, normalized deduplication and new model scoring. Original archive text rights remain separate.
2. **Vlahavas, G. (2024).** *Reddit r/cryptocurrency posts and comments January 2021–December 2022*. [Zenodo record](https://zenodo.org/records/12593440), [DOI](https://doi.org/10.5281/zenodo.12593440), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used the posts and daily BTC-USD OHLCV files, not the comment file. Changes: joined title/body, Bitcoin-keyword filtering, cleaning/deduplication and new scoring. Price provenance is the archived file; the app does not independently certify its upstream venue or fixing time.

Data access date and source hashes are in `data/manifest.json`. Counts and date coverage are calculated from the downloaded files. Raw archives are preserved to make the transformation inspectable. Attribution does not imply endorsement by the original authors.

## Technical blogs: dashboard choices

3. **Plotly (16 October 2026).** [Dashboard Design Mistakes That Mislead Decision Makers](https://plotly.com/blog/dashboard-design-mistakes/). Used as design guidance for visible dates/filter scope, named comparisons, consistent scales and empty-data states. The app shows price history alongside explicit source counts and preserves missing sentiment rather than drawing an invented neutral series. No prose, images or application code copied.
4. **Plotly (30 September 2026).** [Interactive Dashboards vs Static Reports: Which Leads to Better Decisions?](https://plotly.com/blog/interactive-dashboards-vs-static-reports/). Used the overview-to-detail interaction pattern: market pulse → post evidence → model challenge, with matching downloadable tables. No prose, images or application code copied.

## GitHub projects, models and libraries

5. **ProsusAI / FinBERT.** [GitHub project](https://github.com/ProsusAI/finBERT), [original model card](https://huggingface.co/ProsusAI/finbert). Used its pretrained financial-language sentiment model, not its old training notebooks or research results. The original repository supplies its own license. This application does not claim to have trained FinBERT.
6. **Xenova / finbert.** [ONNX conversion](https://huggingface.co/Xenova/finbert). Used `onnx/model_quantized.onnx`, tokenizer and model configuration at the pinned revision `8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20`. Exact hashes are recorded under `models/manifest.json` and copied into the result manifest. The conversion is distinct from the original ProsusAI model.
7. **cjhutto / vaderSentiment.** [GitHub project](https://github.com/cjhutto/vaderSentiment), [MIT license](https://github.com/cjhutto/vaderSentiment/blob/master/LICENSE.txt). Used the installed `vaderSentiment==3.3.2` library and standard compound thresholds ±.05. The lexicon was not retuned on market outcomes.
8. **Microsoft / ONNX Runtime.** [GitHub project](https://github.com/microsoft/onnxruntime), [quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). CPU-only inference with two intra-operation threads. No external inference API.
9. **Plotly.py.** [GitHub project](https://github.com/plotly/plotly.py), [financial-chart examples](https://plotly.com/python/candlestick-charts/), [subplots](https://plotly.com/python/subplots/). Adapted standard public APIs to build OHLC charts and synchronized price/sentiment/count panels with custom colors and units.
10. **scikit-learn.** [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html). Used sparse TF-IDF followed by five-component non-negative matrix factorization for retrospective themes. The code sets a fixed random seed.

All application, data-transformation, aggregation and chart-wrapper code was written for this project. Library code was not vendored. Dependencies retain their licenses, model weights retain their upstream terms, and data retains its original attribution. The neighboring GitHub reference projects were not copied into this addition.
