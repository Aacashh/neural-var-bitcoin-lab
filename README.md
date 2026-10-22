# Market Risk Lab

A small Streamlit project exploring neural networks and market risk. Use the **Workspace** switch to move between two case studies on the same URL:

- **Equity VaR:** a quantile LSTM versus historical, Gaussian and EWMA risk estimates, with breaches, backtests and stress replays.
- **Bitcoin:** archived prices, tweets and Reddit posts, with quantized FinBERT sentiment, VADER, topic exploration and downside-risk charts.

## Run

Python 3.13:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data and scored results are included. Browsing works offline. The optional Bitcoin text demo downloads its pinned 111 MB model on first use and runs on CPU.

To deploy on Streamlit Community Cloud, select `app.py`, use Python 3.13, and install the root `requirements.txt`. No secrets or API keys are needed. `?view=bitcoin` opens the Bitcoin workspace directly.

## Data and notes

Equity returns come from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_5_ind_port.html). Bitcoin uses the [Mendeley tweet archive](https://data.mendeley.com/datasets/8fbdhh72gs/5) and [Zenodo Reddit/price archive](https://zenodo.org/records/12593440), both CC BY 4.0. All charts use frozen historical data.

The default LSTM recorded 20 breaches at 99% confidence over 1,508 test sessions, versus 26 for historical VaR. At 95% it exceeded the target breach rate. Bitcoin sentiment showed little next-day return association in this selected sample. These are retrospective case studies with substantial model and data limitations.

[Equity findings](docs/CASE_STUDY.md) · [Equity methods](docs/METHODOLOGY.md) · [Bitcoin notes](bitcoin/README.md) · [Blogs and GitHub references](bitcoin/REFERENCES.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

For training, audits and tests, install `requirements-dev.txt`, then run `python -m pytest -q`, `python -m risklab audit` and `python -m bitcoin audit`. Rebuild commands and experiment choices are in the methods notes.
