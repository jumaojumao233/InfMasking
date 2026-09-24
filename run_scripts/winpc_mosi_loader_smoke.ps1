param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$BatchSize = 4
)

$ErrorActionPreference = "Stop"
$python = Join-Path $Root ".venv\Scripts\python.exe"
$pythonScript = Join-Path $Root "run_scripts\.winpc_mosi_loader_smoke_runtime.py"

$pythonCode = @'
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from dataset.multibench import MultiBenchDataModule


def tensor_summary(name, values):
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values)!r}")
    print(f"{name.upper()}_MODALITIES={len(values)}")
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}[{index}] must be a Tensor, got {type(value)!r}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}[{index}] contains non-finite values")
        print(f"{name.upper()}_{index}_SHAPE={tuple(value.shape)} DTYPE={value.dtype}")


batch_size = int(os.environ.get("MOSI_SMOKE_BATCH_SIZE", "4"))

ssl = MultiBenchDataModule(
    dataset="mosi",
    model="InfMasking",
    batch_size=batch_size,
    num_workers=0,
    modalities=("vision", "text"),
    task="classification",
    augmentations="drop+noise",
)
print(f"SSL_TRAIN_LEN={len(ssl.train_dataset)}")
print(f"SSL_VAL_LEN={len(ssl.val_dataset)}")
print(f"SSL_TEST_LEN={len(ssl.test_dataset)}")
ssl_batch = next(iter(ssl.train_dataloader()))
if not isinstance(ssl_batch, (list, tuple)) or len(ssl_batch) != 2:
    raise TypeError(f"SSL batch must be a pair of views, got {type(ssl_batch)!r}")
tensor_summary("ssl_view_1", ssl_batch[0])
tensor_summary("ssl_view_2", ssl_batch[1])

sup = MultiBenchDataModule(
    dataset="mosi",
    model="Sup",
    batch_size=batch_size,
    num_workers=0,
    modalities=("vision", "text"),
    task="classification",
)
sup_batch = next(iter(sup.train_dataloader()))
if not isinstance(sup_batch, (list, tuple)) or len(sup_batch) != 2:
    raise TypeError(f"Supervised batch must be a pair of inputs and labels, got {type(sup_batch)!r}")
tensor_summary("sup_input", sup_batch[0])
labels = sup_batch[1]
if not isinstance(labels, torch.Tensor):
    raise TypeError(f"SUP_LABELS must be a Tensor, got {type(labels)!r}")
print(f"SUP_LABELS_SHAPE={tuple(labels.shape)} VALUES={labels.tolist()}")
print("MOSI_LOADER_SMOKE=PASS")
'@

Push-Location $Root
try {
    Set-Content -LiteralPath $pythonScript -Value $pythonCode -Encoding UTF8
    $env:MOSI_SMOKE_BATCH_SIZE = "$BatchSize"
    & $python $pythonScript
    if ($LASTEXITCODE -ne 0) {
        throw "MOSI loader smoke failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item -LiteralPath $pythonScript -Force -ErrorAction SilentlyContinue
    Pop-Location
}
