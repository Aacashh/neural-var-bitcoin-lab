$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $taskPython) {
    & $taskPython -m streamlit run app.py --server.port 8511 --server.address 127.0.0.1
} else {
    python -m streamlit run app.py --server.port 8511 --server.address 127.0.0.1
}
