$ErrorActionPreference = "Stop"

# Post-fix GPU pilot: three paired seeds, Baseline followed by UniGIR.
# Every run uses a fixed 10-epoch budget and evaluates the official test split once.
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$logDir = Join-Path $root "gpu_pilot_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& $python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA is not available in the project Python environment."
}

$hardwareLog = Join-Path $logDir "hardware.txt"
Get-Date -Format o | Set-Content -LiteralPath $hardwareLog
& $python -c "import sys, torch; print(sys.version); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0)); print('memory_gb', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))" | Add-Content -LiteralPath $hardwareLog
"GIT HEAD" | Add-Content -LiteralPath $hardwareLog
git -C $root rev-parse HEAD | Add-Content -LiteralPath $hardwareLog
"GIT STATUS" | Add-Content -LiteralPath $hardwareLog
git -C $root status --short | Add-Content -LiteralPath $hardwareLog
"KEY FILE SHA256" | Add-Content -LiteralPath $hardwareLog
$keyFiles = @(
    "main_trifeatures.py",
    "evaluation/linear_probe.py",
    "losses/infmasking_loss.py",
    "losses/prototype_alignment.py",
    "configs/train_trifeatures.yaml",
    "configs/data/trifeatures.yaml",
    "configs/model/infmasking.yaml",
    "configs/model/unigir.yaml"
) | ForEach-Object { Join-Path $root $_ }
Get-FileHash -Algorithm SHA256 -LiteralPath $keyFiles |
    ForEach-Object { "$($_.Hash)  $($_.Path)" } |
    Add-Content -LiteralPath $hardwareLog

$seeds = @(42, 7, 123)
$experiments = @(
    @{ Name = "baseline"; Model = "infmasking"; Extra = @() },
    @{ Name = "unigir_k128_q1024_a025"; Model = "unigir"; Extra = @(
        "model.model.loss_kwargs.profile_kwargs.num_prototypes=128",
        "model.model.loss_kwargs.profile_kwargs.queue_size=1024",
        "model.model.loss_kwargs.profile_kwargs.loss_weight=0.25",
        "model.model.loss_kwargs.profile_kwargs.cross=false"
    ) }
)

Push-Location $root
try {
    foreach ($seed in $seeds) {
        $common = @(
            "seed=$seed",
            "trainer.strategy=auto",
            "trainer.accelerator=gpu",
            "trainer.devices=1",
            "trainer.deterministic=true",
            "trainer.max_epochs=10",
            "trainer.num_sanity_val_steps=0",
            "+trainer.enable_progress_bar=false",
            "enable_linear_probe=true",
            "probe_frequency=by_fit",
            "enable_early_stopping=false",
            "checkpoint_monitor=null",
            "+data=trifeatures",
            "++data.data_module.biased=true",
            "data.data_module.num_workers=4",
            "data.data_module.max_size=1024",
            "model.model.encoder.embed_dim=256",
            "model.adapters.0.dim_tokens=256",
            "model.adapters.1.dim_tokens=256",
            "++model.model.encoder.num_mask=3",
            "++model.model.encoder.mask_ratio=0.7",
            "optim.lr=3e-4",
            "mode=train"
        )

        foreach ($experiment in $experiments) {
            $runName = "postfix-pilot-S-seed$seed-$($experiment.Name)"
            $log = Join-Path $logDir "$runName.log"
            if ((Test-Path -LiteralPath $log) -and
                (Select-String -LiteralPath $log -Pattern "EXIT_CODE=0" -Quiet)) {
                Write-Host "Skip completed run: $runName"
                continue
            }

            $args = @($common + "+model=$($experiment.Model)" + "+exp_name=$runName") + $experiment.Extra
            "START $(Get-Date -Format o)" | Set-Content -LiteralPath $log
            "ARGS $($args -join ' ')" | Add-Content -LiteralPath $log
            & $python main_trifeatures.py @args *>> $log
            $exitCode = $LASTEXITCODE
            "END $(Get-Date -Format o) EXIT_CODE=$exitCode" | Add-Content -LiteralPath $log
            if ($exitCode -ne 0) {
                throw "Experiment $runName failed with exit code $exitCode. See $log"
            }
        }
    }
}
finally {
    Pop-Location
}
