"""Generate the short findings note from saved artifacts, without fitting anything."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from risklab.data import load_data

metrics=pd.read_csv(ROOT/"results"/"metrics.csv")
manifest=json.loads((ROOT/"results"/"manifest.json").read_text())
frame,source=load_data()
rows=["# A small neural-network market-risk case study", "",
      "**Question:** Can an LSTM improve a one-day loss-quantile forecast for a fixed, daily rebalanced five-industry equity portfolio?", "",
      "The public snapshot contains 6,791 sessions through 2025. The experiment uses 4,279 training targets through 2016, 754 validation targets in 2017–2019, and 1,508 untouched test targets in 2020–2025. All models forecast the same next-session loss. The default neural seed is 7, chosen in the protocol before test evaluation.", ""]
for q in [.99,.95]:
    rows += [f"## {q:.0%} one-day VaR", "", "| Model | Breaches | Breach rate | Expected | Pinball (bps) | Kupiec p | Independence p |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    selected=metrics.loc[(metrics["quantile"]==q)&((metrics.model!="LSTM")|(metrics.seed==7))]
    for _,r in selected.iterrows():
        rows.append(f"| {r.model} | {int(r.exceptions)} | {r.exception_rate:.2%} | {r.expected:.2f} | {r.pinball*10000:.3f} | {r.kupiec_p:.4g} | {r.independence_p:.4g} |")
    rows.append("")
rows += ["## What I learned", "",
         "At 99%, the default LSTM has the lowest descriptive pinball score among the four models and fewer breaches, closer to the target than the benchmarks. Its 20 breaches still exceed the expected 15.08. Non-rejection of coverage or independence does not establish that it is safe or correctly specified.", "",
         "At 95%, the default LSTM's 88 exceptions (5.84%) exceed the 5% target and the historical benchmark's 76. All three neural seeds show excess 95% breaches. The historical benchmark's overall frequency is close to target, but its exceptions cluster. This illustrates why frequency, clustering and quantile loss answer different questions.", "",
         "The LSTM is frozen after validation while its input sequences update. Baselines update their histories daily. Results are limited to this portfolio and time period. No test-set tuning, production suitability or statistically significant model superiority is claimed.", "",
         "## Initialization sensitivity", "", "| Seed | 95% breaches | 95% pinball (bps) | 99% breaches | 99% pinball (bps) |",
         "|---:|---:|---:|---:|---:|"]
for seed in [7,19,42]:
    part=metrics.loc[(metrics.model=="LSTM")&(metrics.seed==seed)].set_index("quantile")
    lo,hi=part.loc[.95],part.loc[.99]
    rows.append(f"| {seed} | {int(lo.exceptions)} | {lo.pinball*10000:.3f} | {int(hi.exceptions)} | {hi.pinball*10000:.3f} |")
rows += ["", "These seeds measure sensitivity to optimization, not uncertainty over future market regimes. Comparisons and p-values are descriptive and unadjusted for multiple testing.", "",
         "## Historical stress replay", "", "| Window (after start close) | Portfolio return | Change on initial $1m | Worst one-day return |", "|---|---:|---:|---:|"]
for label,start,end in [("Financial crisis","2007-10-09","2009-03-09"),("COVID sell-off","2020-02-19","2020-03-23"),("2022","2021-12-31","2022-12-30"),("Spring 2025","2025-02-19","2025-04-08")]:
    p=frame.loc[(frame.index>start)&(frame.index<=end)].mean(axis=1)
    change=(1+p).prod()-1
    rows.append(f"| {label}: {start} to {end} | {change:.2%} | ${change*1000000:,.0f} | {p.min():.2%} |")
rows += ["", "Stress windows are retrospective demonstrations. Financial-crisis observations are in training, so their replay is not held-out neural evidence. Compounded stress P&L differs from the constant-notional daily losses used in VaR backtesting.", "",
         "## Reproduction", "", f"All three CPU fits and benchmark exports took **{manifest['total_seconds']:.1f} seconds** in the recorded environment, excluding download and app rendering. Best validation epochs were " + ", ".join(f"seed {r['seed']}: {r['best_epoch']}" for r in manifest["runtime"]) + ".", "",
         "`python -m risklab audit` verifies the raw-to-curated transform, source/code checksums, complete held-out forecast calendars, prior-session origins, actual portfolio losses, baseline replays, the train-only scaler, all three saved neural inference replays, noncrossing quantiles and metric recomputation.", "",
         f"Dataset: [Kenneth French five-industry returns]({source['details']}). See [methodology](METHODOLOGY.md) for equations and limits."]
(ROOT/"docs"/"CASE_STUDY.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
print(ROOT/"docs"/"CASE_STUDY.md")
