"""Replay the full corpus after setup changes; preserve both result manifests."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import bitcoin.pipeline as pipeline

old=pipeline.RESULTS
replayed=old.with_name("results_replayed")
archive=old.with_name("results_initial_transport")
for path in [old,replayed,archive]:
    assert path.resolve().is_relative_to(root.resolve())
if replayed.exists() or archive.exists():
    raise FileExistsError("Replay evidence exists; inspect it before another run")
pipeline.RESULTS=replayed
run=pipeline.build()
before=pd.read_csv(old/"posts_scored.csv")
after=pd.read_csv(replayed/"posts_scored.csv")
assert before.post_id.tolist()==after.post_id.tolist()
np.testing.assert_allclose(before[["positive","negative","neutral","finbert_score","vader_score"]],
                           after[["positive","negative","neutral","finbert_score","vader_score"]],atol=1e-7)
assert before.finbert_label.tolist()==after.finbert_label.tolist()
assert before.topic.tolist()==after.topic.tolist()
old.rename(archive)
replayed.rename(old)
pipeline.RESULTS=old
receipt=dict(status="passed",all_posts_replayed=len(after),
             original_manifest=str(archive.relative_to(root)/"manifest.json"),
             final_code_sha256=run["code_sha256"],
             reason="Model download restoration was improved after the first score build; numerical inference was unchanged.",
             audit=pipeline.audit())
(old/"replay_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print(json.dumps(receipt,indent=2))
