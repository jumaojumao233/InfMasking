$ErrorActionPreference = "Continue"

# Compare the baseline and the two shortlisted UniGIR variants with probing.
# Keep the runs sequential because this machine has no GPU.
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$logDir = Join-Path $root "probe_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$common = @(
    "seed=42",
    "trainer.strategy=auto",
    "trainer.accelerator=cpu",
    "trainer.devices=1",
    "trainer.max_epochs=1",
    "trainer.num_sanity_val_steps=0",
    "+trainer.enable_progress_bar=false",
    "enable_linear_probe=true",
    "+data=trifeatures",
    "++data.data_module.biased=true",
    "data.data_module.num_workers=0",
    "data.data_module.max_size=256",
    "model.model.encoder.embed_dim=256",
    "model.adapters.0.dim_tokens=256",
    "model.adapters.1.dim_tokens=256",
    "++model.model.encoder.num_mask=3",
    "++model.model.encoder.mask_ratio=0.7",
    "optim.lr=3e-4",
    "mode=train"
)

$experiments = @(
    @{ Name = "baseline"; Model = "infmasking"; Extra = @() },
    @{ Name = "profile_k128_q1024_a025"; Model = "unigir"; Extra = @(
        "model.model.loss_kwargs.profile_kwargs.num_prototypes=128",
        "model.model.loss_kwargs.profile_kwargs.queue_size=1024",
        "model.model.loss_kwargs.profile_kwargs.loss_weight=0.25"
    ) },
    @{ Name = "profile_k128_q1024_a025_cross"; Model = "unigir"; Extra = @(
        "model.model.loss_kwargs.profile_kwargs.num_prototypes=128",
        "model.model.loss_kwargs.profile_kwargs.queue_size=1024",
        "model.model.loss_kwargs.profile_kwargs.loss_weight=0.25",
        "model.model.loss_kwargs.profile_kwargs.cross=true"
    ) }
)

Push-Location $root
try {
    foreach ($experiment in $experiments) {
        $args = @($common + "+model=$($experiment.Model)" + "+exp_name=cpu-probe-$($experiment.Name)") + $experiment.Extra
        $log = Join-Path $logDir "$($experiment.Name).log"
        Write-Host "Starting $($experiment.Name)"
        & $python main_trifeatures.py @args *> $log
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $($experiment.Name) failed with exit code $LASTEXITCODE. See $log"
        }
        Write-Host "Finished $($experiment.Name)"
    }
}
finally {
    Pop-Location
}
