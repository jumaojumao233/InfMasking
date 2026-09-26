param(
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$DiagnosticScript = "run_scripts\diagnose_fusion_relation_structure.py",
    [int]$ModelSeed = 42,
    [int]$BatchSize = 64,
    [int]$MaxSamples = 256,
    [int]$TrainSamples = 512,
    [int]$TestSamples = 256,
    [double]$RidgeAlpha = 1.0,
    [string]$MaskRatios = "",
    [string]$MaskSeeds = "",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"
$logRoot = Join-Path $root "winpc_relation_diag_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$stdoutLog = Join-Path $logRoot "$RunName.stdout.log"
$stderrLog = Join-Path $logRoot "$RunName.stderr.log"
$statusLog = Join-Path $logRoot "$RunName.status.log"
$scriptPath = if ([IO.Path]::IsPathRooted($DiagnosticScript)) {
    $DiagnosticScript
}
else {
    Join-Path $root $DiagnosticScript
}

"START $(Get-Date -Format o)" | Set-Content -LiteralPath $statusLog
"RUN_NAME=$RunName MODEL_SEED=$ModelSeed BATCH_SIZE=$BatchSize MAX_SAMPLES=$MaxSamples DEVICE=$Device" | Add-Content -LiteralPath $statusLog
"TRAIN_SAMPLES=$TrainSamples TEST_SAMPLES=$TestSamples RIDGE_ALPHA=$RidgeAlpha MASK_RATIOS=$MaskRatios MASK_SEEDS=$MaskSeeds" | Add-Content -LiteralPath $statusLog
"CHECKPOINT=$Checkpoint" | Add-Content -LiteralPath $statusLog
"DATA_ROOT=$DataRoot" | Add-Content -LiteralPath $statusLog
"OUTPUT_DIR=$OutputDir" | Add-Content -LiteralPath $statusLog
"STDOUT=$stdoutLog" | Add-Content -LiteralPath $statusLog
"STDERR=$stderrLog" | Add-Content -LiteralPath $statusLog

$args = @(
    $scriptPath,
    "--checkpoint", $Checkpoint,
    "--data-root", $DataRoot,
    "--output-dir", $OutputDir,
    "--model-seed", $ModelSeed,
    "--batch-size", $BatchSize,
    "--device", $Device
)

$parsedMaskRatios = @()
if (-not [string]::IsNullOrWhiteSpace($MaskRatios)) {
    $parsedMaskRatios = @($MaskRatios -split '[,\s]+' | Where-Object { $_ -ne "" } |
        ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) })
}
$parsedMaskSeeds = @()
if (-not [string]::IsNullOrWhiteSpace($MaskSeeds)) {
    $parsedMaskSeeds = @($MaskSeeds -split '[,\s]+' | Where-Object { $_ -ne "" } |
        ForEach-Object { [int]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) })
}

if ($scriptPath -match "diagnose_nonadditive_residual_structure\.py$") {
    $args += @("--train-samples", $TrainSamples, "--test-samples", $TestSamples, "--ridge-alpha", $RidgeAlpha)
}
else {
    $args += @("--max-samples", $MaxSamples)
}

if ($parsedMaskRatios.Count -gt 0) {
    $args += "--mask-ratios"
    $args += @($parsedMaskRatios)
}
if ($parsedMaskSeeds.Count -gt 0) {
    $args += "--mask-seeds"
    $args += @($parsedMaskSeeds)
}

$exitCode = 1
Push-Location $root
try {
    & $python @args 1> $stdoutLog 2> $stderrLog
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
