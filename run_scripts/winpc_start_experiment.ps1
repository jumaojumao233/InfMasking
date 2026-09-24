param(
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [ValidateSet("baseline", "unigir")]
    [string]$Method,
    [Parameter(Mandatory = $true)]
    [int]$Seed,
    [int]$GpuIndex = 0,
    [int]$MaxEpochs = 10,
    [int]$MaxSize = 1024,
    [int]$PairSeed = 42,
    [string]$DataRoot = "",
    [string]$ResumeCkptPath = "",
    [int]$QueueSize = 1024,
    [switch]$EnableLinearProbe,
    [switch]$ShuffleProfile,
    [switch]$CudaLaunchBlocking
)

$ErrorActionPreference = "Continue"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
$runRoot = Join-Path $root "winpc_runs"
$logRoot = Join-Path $root "winpc_logs"
New-Item -ItemType Directory -Force -Path $runRoot, $logRoot | Out-Null

$stdoutLog = Join-Path $logRoot "$RunName.stdout.log"
$stderrLog = Join-Path $logRoot "$RunName.stderr.log"
$statusLog = Join-Path $logRoot "$RunName.status.log"
$modelConfig = if ($Method -eq "baseline") { "infmasking" } else { "unigir" }

$args = @(
    "seed=$Seed",
    "trainer.strategy=auto",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.deterministic=true",
    "trainer.max_epochs=$MaxEpochs",
    "trainer.num_sanity_val_steps=0",
    "+trainer.enable_progress_bar=false",
    "enable_linear_probe=$($EnableLinearProbe.IsPresent.ToString().ToLower())",
    "probe_frequency=by_fit",
    "enable_early_stopping=false",
    "checkpoint_monitor=null",
    "+data=trifeatures",
    "++data.data_module.biased=true",
    "data.data_module.num_workers=0",
    "data.data_module.max_size=$MaxSize",
    "data.data_module.seed=$PairSeed",
    "model.model.encoder.embed_dim=256",
    "model.adapters.0.dim_tokens=256",
    "model.adapters.1.dim_tokens=256",
    "++model.model.encoder.num_mask=3",
    "++model.model.encoder.mask_ratio=0.7",
    "optim.lr=3e-4",
    "trainer.default_root_dir=$runRoot",
    "+model=$modelConfig",
    "+exp_name=$RunName",
    "mode=train"
)

if ($DataRoot -ne "") {
    $args += "data.data_module.data_root=$DataRoot"
}
if ($ResumeCkptPath -ne "") {
    $args += "resume_ckpt_path=$ResumeCkptPath"
}

if ($Method -eq "unigir") {
    $args += @(
        "model.model.loss_kwargs.profile_kwargs.num_prototypes=128",
        "model.model.loss_kwargs.profile_kwargs.queue_size=$QueueSize",
        "model.model.loss_kwargs.profile_kwargs.loss_weight=0.25",
        "model.model.loss_kwargs.profile_kwargs.cross=false",
        "model.model.loss_kwargs.profile_kwargs.shuffle_targets=$($ShuffleProfile.IsPresent.ToString().ToLower())"
    )
}

if ($CudaLaunchBlocking) {
    $env:CUDA_LAUNCH_BLOCKING = "1"
}

$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
"START $(Get-Date -Format o)" | Set-Content -LiteralPath $statusLog
"METHOD=$Method SEED=$Seed PAIR_SEED=$PairSeed DATA_ROOT=$DataRoot GPU_INDEX=$GpuIndex MAX_EPOCHS=$MaxEpochs MAX_SIZE=$MaxSize QUEUE_SIZE=$QueueSize LINEAR_PROBE=$($EnableLinearProbe.IsPresent) SHUFFLE_PROFILE=$($ShuffleProfile.IsPresent) CUDA_LAUNCH_BLOCKING=$($CudaLaunchBlocking.IsPresent)" | Add-Content -LiteralPath $statusLog
"STDOUT=$stdoutLog" | Add-Content -LiteralPath $statusLog
"STDERR=$stderrLog" | Add-Content -LiteralPath $statusLog
"ARGS=$($args -join ' ')" | Add-Content -LiteralPath $statusLog

$exitCode = 1
Push-Location $root
try {
    & $python "main_trifeatures.py" @args 1> $stdoutLog 2> $stderrLog
    $exitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrLog
}
finally {
    Pop-Location
}

"END $(Get-Date -Format o) EXIT_CODE=$exitCode" | Add-Content -LiteralPath $statusLog
exit $exitCode
