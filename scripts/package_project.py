"""Small portable demo: includes frozen evidence; excludes environments and optional large weights."""
from pathlib import Path
import subprocess
import zipfile

root=Path(__file__).resolve().parents[1]
archive=root.parent/"Neural_VaR_and_Bitcoin_Labs.zip"
files=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"],cwd=root).decode().split("\0")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as out:
    for name in sorted(set(files)-{""}):
        file=root/name
        if not file.is_file():
            continue
        relative=file.relative_to(root)
        out.write(file,Path("neural_var_lab")/relative)
with zipfile.ZipFile(archive) as saved:
    assert saved.testzip() is None
    names=saved.namelist()
    assert "neural_var_lab/bitcoin/results/posts_scored.csv" in names
    assert "neural_var_lab/results/forecasts.csv" in names
    assert not any("/.venv/" in p or p.endswith(".onnx") for p in names)
print(f"{archive} · {archive.stat().st_size/1e6:.2f} MB · {len(names)} files")
