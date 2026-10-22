$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'Use Workspace > Bitcoin in the shared dashboard.'
& (Join-Path $PSScriptRoot 'Start-Lab.ps1')
