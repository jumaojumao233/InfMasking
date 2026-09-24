$ErrorActionPreference = "Continue"

# Compare the selected UniGIR direction with and without a feature queue.
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$logDir = Join-Path $root "seed7_noqueue_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$args = @(
    "seed=7",
    "trainer.strategy=auto",
    "trainer.accelerator=cpu",
    "trainer.devices=1",
    "trainer.max_epochs=4",
    "trainer.num_sanity_val_steps=0",
    "+trainer.enable_progress_bar=false",
    "enable_linear_probe=true",
    "+data=trifeatures",
    "++data.data_module.biased=true",
    "data.data_module.num_workers=0",
    "data.data_module.max_size=1024",
    "model.model.encoder.embed_dim=256",
    "model.adapters.0.dim_tokens=256",
    "model.adapters.1.dim_tokens=256",
    "++model.model.encoder.num_mask=3",
    "++model.model.encoder.mask_ratio=0.7",
    "optim.lr=3e-4",
    "mode=train",
    "+model=unigir",
    "+exp_name=seed7-S-unigir_k128_q0_a025",
    "model.model.loss_kwargs.profile_kwargs.num_prototypes=128",
    "model.model.loss_kwargs.profile_kwargs.queue_size=0",
    "model.model.loss_kwargs.profile_kwargs.loss_weight=0.25",
    "model.model.loss_kwargs.profile_kwargs.cross=false"
)

Push-Location $root
try {
    $log = Join-Path $logDir "unigir_k128_q0_a025.log"
    & $python main_trifeatures.py @args *> $log
    $exitCode = $LASTEXITCODE
    Add-Content -Path $log -Value "END $(Get-Date -Format o) EXIT_CODE=$exitCode"
    if ($exitCode -ne 0) {
        throw "Experiment unigir_k128_q0_a025 failed with exit code $exitCode. See $log"
    }
}
finally {
    Pop-Location
}
