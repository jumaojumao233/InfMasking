param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Proxy = ""
)

$ErrorActionPreference = "Stop"
$python = Join-Path $Root ".venv\Scripts\python.exe"
$dataPath = Join-Path $Root "dataset\data\mosi\mosi_data.pkl"
$pythonScript = Join-Path $Root "run_scripts\.winpc_prepare_mosi_runtime.py"

if (-not [string]::IsNullOrWhiteSpace($Proxy)) {
    $env:HTTP_PROXY = $Proxy
    $env:HTTPS_PROXY = $Proxy
    $env:ALL_PROXY = $Proxy
    Write-Output "PROXY_ENABLED=True"
}
else {
    Write-Output "PROXY_ENABLED=False"
}

$pythonCode = @'
import hashlib
import os
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())

import numpy as np
from dataset.affect.get_data import Affect

path = os.path.join("dataset", "data", "mosi", "mosi_data.pkl")
for split in ("train", "val", "test"):
    dataset = Affect(path, "mosi", split=split, modalities=("vision", "text"))
    labels = np.asarray(dataset.dataset["labels"]).reshape(-1)
    counts = Counter(int(value > 0) for value in labels)
    print(f"SPLIT={split} N={len(dataset)} LABEL_COUNTS={dict(sorted(counts.items()))}")

with open(path, "rb") as handle:
    digest = hashlib.sha256(handle.read()).hexdigest()
print(f"DATA_PATH={os.path.abspath(path)}")
print(f"DATA_SIZE={os.path.getsize(path)}")
print(f"DATA_SHA256={digest}")
'@

Push-Location $Root
try {
    Set-Content -LiteralPath $pythonScript -Value $pythonCode -Encoding UTF8
    & $python $pythonScript
    if ($LASTEXITCODE -ne 0) {
        throw "MOSI preparation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item -LiteralPath $pythonScript -Force -ErrorAction SilentlyContinue
    Pop-Location
}

if (-not (Test-Path -LiteralPath $dataPath)) {
    throw "MOSI data file was not created: $dataPath"
}

$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $dataPath
$item = Get-Item -LiteralPath $dataPath
Write-Output "CHECKED_PATH=$($item.FullName)"
Write-Output "CHECKED_SIZE=$($item.Length)"
Write-Output "CHECKED_SHA256=$($hash.Hash.ToLower())"
