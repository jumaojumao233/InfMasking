param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RunName = "winpc-G3-mosi-unigir-smoke",
    [ValidateSet("baseline", "unigir")]
    [string]$Method = "unigir",
    [int]$Seed = 42,
    [int]$GpuIndex = 0,
    [int]$MaxEpochs = 1,
    [int]$BatchSize = 32,
    [ValidateSet("by_epoch", "by_fit")]
    [string]$ProbeFrequency = "by_fit"
)

$ErrorActionPreference = "Continue"
$env:HYDRA_FULL_ERROR = "1"
$python = Join-Path $Root ".venv\Scripts\python.exe"
$runRoot = Join-Path $Root "winpc_mosi_runs"
$logRoot = Join-Path $Root "winpc_mosi_logs"
New-Item -ItemType Directory -Force -Path $runRoot, $logRoot | Out-Null

$stdoutLog = Join-Path $logRoot "$RunName.stdout.log"
$stderrLog = Join-Path $logRoot "$RunName.stderr.log"
$statusLog = Join-Path $logRoot "$RunName.status.log"
$modelConfig = if ($Method -eq "baseline") { "InfMasking" } else { "unigir" }

$args = @(
    "model=$modelConfig",
    "data.data_module.dataset=mosi",
    "seed=$Seed",
    "mode=train",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.deterministic=true",
    "trainer.max_epochs=$MaxEpochs",
    "+trainer.num_sanity_val_steps=0",
    "trainer.default_root_dir=$runRoot",
    "+trainer.enable_progress_bar=false",
    "data.data_module.batch_size=$BatchSize",
    "data.data_module.num_workers=0",
    "linear_probing.frequency=$ProbeFrequency",
    "linear_probing.fastsearch=true",
    "+linear_probing.max_iter=10",
    "linear_probing.use_sklearn=true",
    "+exp_name=$RunName"
)

$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
"START $(Get-Date -Format o)" | Set-Content -LiteralPath $statusLog
"METHOD=$Method DATASET=mosi SEED=$Seed GPU_INDEX=$GpuIndex MAX_EPOCHS=$MaxEpochs BATCH_SIZE=$BatchSize PROBE_FREQUENCY=$ProbeFrequency" | Add-Content -LiteralPath $statusLog
"STDOUT=$stdoutLog" | Add-Content -LiteralPath $statusLog
"STDERR=$stderrLog" | Add-Content -LiteralPath $statusLog
"ARGS=$($args -join ' ')" | Add-Content -LiteralPath $statusLog

$exitCode = 1
Push-Location $Root
try {
    & $python "main_multibench.py" @args 1> $stdoutLog 2> $stderrLog
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
