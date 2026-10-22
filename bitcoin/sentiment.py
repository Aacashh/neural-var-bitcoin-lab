"""CPU-only int8 FinBERT inference. Imports stay lazy for the cached dashboard."""
import json
import hashlib
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import requests
from scipy.special import softmax

from .config import MODELS, MODEL_ID, MODEL_REVISION, MAX_TOKENS, BATCH_SIZE
from .data import digest, clean_text


def download_model():
    MODELS.mkdir(parents=True,exist_ok=True)
    previous=json.loads((MODELS/"manifest.json").read_text(encoding="utf-8")) if (MODELS/"manifest.json").exists() else None
    if previous and (previous["model_id"]!=MODEL_ID or previous["revision"]!=MODEL_REVISION):
        raise ValueError("Existing model manifest has another identity")
    revision=MODEL_REVISION
    names=["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","onnx/model_quantized.onnx"]
    # Reviewed official repository blob identities at the pinned revision.
    identities={"config.json":("git","03a8cbf103c162c52e162efca395d5dc636e1d56"),
                "tokenizer.json":("git","688882a79f44442ddc1f60d70334a7ff5df0fb47"),
                "tokenizer_config.json":("git","37fca74771bc76a8e01178ce3a6055a0995f8093"),
                "special_tokens_map.json":("git","a8b3208c2884c4efb86e49300fdd3dc877220cdf"),
                "onnx/model_quantized.onnx":("sha256","56cd19c7068ca5670c326816845bb7d7edda508b68a228697a1460d0e821ce76")}
    def valid(path,name):
        if not path.exists():
            return False
        kind,expected=identities[name]
        data=path.read_bytes()
        actual=hashlib.sha256(data).hexdigest() if kind=="sha256" else hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
        return actual==expected
    checksums={}
    for name in names:
        target=MODELS/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not valid(target,name):
            url=f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}?download=true"
            partial=target.with_suffix(target.suffix+".part")
            print(f"Downloading {name}...",flush=True)
            # Windows' system TLS transport is more reliable on this host than
            # the Python clients. Both transports retain certificate validation.
            curl=shutil.which("curl.exe") if os.name=="nt" else None
            if curl:
                subprocess.run([curl,"--fail","--silent","--show-error","--location","--connect-timeout","20",
                                "--max-time","300","--retry","2","--retry-delay","2","--output",str(partial),url],check=True)
            else:
                with requests.get(url,stream=True,timeout=90) as response:
                    response.raise_for_status()
                    with partial.open("wb") as file:
                        for chunk in response.iter_content(65536):
                            file.write(chunk)
            if not valid(partial,name):
                raise ValueError(f"Official model blob identity mismatch: {name}")
            os.replace(partial,target)
        checksums[name]=digest(target)
        if previous and checksums[name]!=previous["checksums"][name]:
            raise ValueError(f"Restored model file differs from frozen manifest: {name}")
        print(f"Downloaded {name}: {target.stat().st_size/1e6:.1f} MB",flush=True)
    manifest=previous or dict(model_id=MODEL_ID,revision=revision,downloaded_utc=datetime.now(timezone.utc).isoformat(),
                  original_model="ProsusAI/finbert",format="quantized ONNX; CPUExecutionProvider",
                  checksums=checksums)
    (MODELS/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


class SentimentEngine:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.manifest=json.loads((MODELS/"manifest.json").read_text(encoding="utf-8"))
        for name,expected in self.manifest["checksums"].items():
            if digest(MODELS/name)!=expected:
                raise ValueError(f"Model checksum mismatch: {name}")
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        options.log_severity_level=3
        self.session=ort.InferenceSession(str(MODELS/"onnx"/"model_quantized.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
        self.tokenizer=Tokenizer.from_file(str(MODELS/"tokenizer.json"))
        self.raw_tokenizer=Tokenizer.from_file(str(MODELS/"tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0,pad_token="[PAD]")
        config=json.loads((MODELS/"config.json").read_text())
        self.labels=[config["id2label"][str(i)].lower() for i in range(3)]
        self.vader=SentimentIntensityAnalyzer()

    def predict(self,texts,progress=False):
        cleaned=[clean_text(t) for t in texts]
        outputs=[]
        started=time.perf_counter()
        for offset in range(0,len(cleaned),BATCH_SIZE):
            batch=cleaned[offset:offset+BATCH_SIZE]
            raw=self.raw_tokenizer.encode_batch(batch)
            encoded=self.tokenizer.encode_batch(batch)
            tensors={"input_ids":np.asarray([e.ids for e in encoded],dtype=np.int64),
                     "attention_mask":np.asarray([e.attention_mask for e in encoded],dtype=np.int64),
                     "token_type_ids":np.asarray([e.type_ids for e in encoded],dtype=np.int64)}
            feed={x.name:tensors[x.name] for x in self.session.get_inputs()}
            probs=softmax(self.session.run(None,feed)[0],axis=1)
            for text,p,original in zip(batch,probs,raw):
                mapped=dict(zip(self.labels,p))
                vader=self.vader.polarity_scores(text)["compound"]
                fin_label=self.labels[int(p.argmax())]
                vader_label="positive" if vader>=.05 else "negative" if vader<=-.05 else "neutral"
                outputs.append(dict(positive=float(mapped["positive"]),negative=float(mapped["negative"]),
                                    neutral=float(mapped["neutral"]),finbert_score=float(mapped["positive"]-mapped["negative"]),
                                    finbert_label=fin_label,confidence=float(p.max()),
                                    entropy=float(-(p*np.log(np.maximum(p,1e-12))).sum()/np.log(3)),
                                    vader_score=float(vader),vader_label=vader_label,disagreement=fin_label!=vader_label,
                                    token_count=len(original.ids),truncated=len(original.ids)>MAX_TOKENS))
            if progress and (offset//BATCH_SIZE)%20==0:
                print(f"Scored {min(offset+BATCH_SIZE,len(cleaned))}/{len(cleaned)} posts · {time.perf_counter()-started:.1f}s",flush=True)
        return outputs
