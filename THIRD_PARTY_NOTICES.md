# Sources and reuse

The Bitcoin extension uses separate CC BY research archives, pretrained FinBERT/ONNX weights and the installed VADER library. See [Bitcoin references](bitcoin/REFERENCES.md) for data attribution, model revision, upstream GitHub projects, technical blogs and the exact reuse boundaries.

Application and numerical code were written for this focused project. No source files from the neighboring reference repositories or the existing `finance_bro` application were copied. Standard historical/EWMA risk methods and PyTorch's LSTM are used.

Market returns are from the Kenneth R. French Data Library. The source page states copyright Eugene F. Fama and Kenneth R. French. Provider rights in the data are separate from this project's code; public access does not place the data in the public domain. The raw ZIP, official URLs and checksums are retained in `data/manifest.json`. This local research demo bundles its frozen research slice; check applicable provider permissions before broader redistribution or public hosting.

Libraries retain their respective licenses. The pinned requirements identify Streamlit, NumPy, pandas, SciPy, Plotly, scikit-learn, Requests, PyTorch and pytest. No bank code, internal positions or private market data are used.
