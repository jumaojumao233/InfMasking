<#
    Run a read-only checkpoint evaluation on winpc.

    This script never calls trainer.fit. It runs main_trifeatures.py in
    mode=test, exports the selected probe predictions, and sends a mail at
    start, every 15 minutes, and completion.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [ValidateSet("baseline", "unigir")]
    [string]$Method,
    [Parameter(Mandatory = $true)]
    [int]$Seed,
    [Parameter(Mandatory = $true)]
    [string]$CheckpointPath,
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [int]$GpuIndex = 0,
    [int]$NotifyIntervalSeconds = 900,
    [switch]$ProfileOnly,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$notifyScript = Join-Path $ProjectRoot "run_scripts\winpc_g4_notify.py"
$mailEnvScript = Join-Path $env:USERPROFILE "secrets\InfMasking\mail.env.ps1"
$logRoot = Join-Path $ProjectRoot "winpc_logs"
$stdoutLog = Join-Path $logRoot "$RunName.eval.stdout.log"
$stderrLog = Join-Path $logRoot "$RunName.eval.stderr.log"
$statusLog = Join-Path $logRoot "$RunName.eval.status.log"
$bodyPath = Join-Path $logRoot "$RunName.eval.email.body.txt"
$mailErrorPath = Join-Path $logRoot "$RunName.eval.mail.stderr.log"
$predictionRoot = Join-Path $OutputRoot "predictions"
$modelConfig = if ($Method -eq "baseline") { "infmasking" } else { "unigir" }

New-Item -ItemType Directory -Force -Path $logRoot, $OutputRoot, $predictionRoot | Out-Null

function Send-EvalNotification {
    param([string]$Action, [string]$Detail)
    if (-not (Test-Path -LiteralPath $mailEnvScript)) { return "MAIL_CONFIG_MISSING" }
    if (-not (Test-Path -LiteralPath $notifyScript)) { return "MAIL_SCRIPT_MISSING" }
    if (-not (Test-Path -LiteralPath $python)) { return "MAIL_PYTHON_MISSING" }
    try {
        . $mailEnvScript
        $body = @(
            "Timestamp: $(Get-Date -Format o)",
            "Host: $env:COMPUTERNAME",
            "Evaluation run: $RunName",
            "Method: $Method",
            "Seed: $Seed",
            "Action: $Action",
            "Detail: $Detail",
            "Checkpoint: $CheckpointPath",
            "Prediction directory: $predictionRoot"
        )
        $body | Set-Content -LiteralPath $bodyPath -Encoding utf8
        & $python $notifyScript --subject "[InfMasking checkpoint eval][$Action] $RunName" --body-file $bodyPath 2>> $mailErrorPath | Out-Null
        if ($LASTEXITCODE -eq 0) { return "EMAIL_SENT" }
        return "EMAIL_FAILED_EXIT_$LASTEXITCODE"
    }
    catch { return "EMAIL_FAILED: $($_.Exception.Message)" }
}

if (-not (Test-Path -LiteralPath $CheckpointPath)) {
    "$(Get-Date -Format o) ACTION=STOP DETAIL=checkpoint missing: $CheckpointPath" | Set-Content -LiteralPath $statusLog -Encoding utf8
    Send-EvalNotification -Action "STOP" -Detail "checkpoint missing: $CheckpointPath" | Out-Null
    exit 2
}

$args = @(
    "main_trifeatures.py",
    "seed=$Seed",
    "mode=test",
    "+ckpt_path=$CheckpointPath",
    "+model=$modelConfig",
    "+exp_name=$RunName",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.default_root_dir=$OutputRoot",
    "test_root_dir=$OutputRoot",
    "trainer.inference_mode=false",
    "trainer.num_sanity_val_steps=0",
    "+trainer.enable_progress_bar=false",
    "+data=trifeatures",
    "++data.data_module.fixed_eval_transform=true",
    "++data.data_module.data_root=$DataRoot",
    "++data.data_module.num_workers=0"
)

if ($ProfileOnly) {
    $args += "enable_linear_probe=false"
}
else {
    $args += @(
        "enable_linear_probe=true",
        "probe_frequency=by_fit",
        "probe_names=[unique2]",
        "export_predictions_dir=$predictionRoot"
    )
}

$args += @(
    "model.model.encoder.embed_dim=256",
    "model.adapters.0.dim_tokens=256",
    "model.adapters.1.dim_tokens=256",
    "++model.model.encoder.num_mask=3",
    "++model.model.encoder.mask_ratio=0.7"
)
if ($Method -eq "unigir") {
    $args += @(
        "model.model.loss_kwargs.profile_kwargs.num_prototypes=128",
        "model.model.loss_kwargs.profile_kwargs.queue_size=1024",
        "model.model.loss_kwargs.profile_kwargs.loss_weight=0.25",
        "model.model.loss_kwargs.profile_kwargs.cross=false",
        "model.model.loss_kwargs.profile_kwargs.shuffle_targets=false"
    )
}

$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
"START $(Get-Date -Format o)" | Set-Content -LiteralPath $statusLog -Encoding utf8
"METHOD=$Method SEED=$Seed CHECKPOINT=$CheckpointPath DATA_ROOT=$DataRoot OUTPUT_ROOT=$OutputRoot GPU_INDEX=$GpuIndex" | Add-Content -LiteralPath $statusLog
"ARGS=$($args -join ' ')" | Add-Content -LiteralPath $statusLog
$startMail = Send-EvalNotification -Action "START" -Detail "evaluation process requested; no training will run"
"START_NOTIFICATION=$startMail" | Add-Content -LiteralPath $statusLog

Push-Location $ProjectRoot
try {
    $process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru -WindowStyle Hidden
    $lastMailAt = Get-Date
    while (-not $process.HasExited) {
        if (-not $process.WaitForExit(60000)) {
            $elapsed = (Get-Date) - $lastMailAt
            if ($elapsed.TotalSeconds -ge $NotifyIntervalSeconds) {
                $mail = Send-EvalNotification -Action "WAIT" -Detail "evaluation still running; process id=$($process.Id)"
                "WAIT $(Get-Date -Format o) PID=$($process.Id) NOTIFICATION=$mail" | Add-Content -LiteralPath $statusLog -Encoding utf8
                $lastMailAt = Get-Date
            }
        }
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = [int]$process.ExitCode
    $stderrText = if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Raw } else { "" }
    if ($exitCode -eq 0 -and $stderrText -match "Could not override|Traceback \(most recent call last\)|CUDA error|Exception:") {
        $exitCode = 1
    }
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrLog -Encoding utf8
    $exitCode = 1
}
finally {
    Pop-Location
}

"END $(Get-Date -Format o) EXIT_CODE=$exitCode" | Add-Content -LiteralPath $statusLog -Encoding utf8
$action = if ($exitCode -eq 0) { "COMPLETE" } else { "FAILED" }
$detail = if ($exitCode -eq 0) { "evaluation completed" } else { "evaluation exit code $exitCode" }
$endMail = Send-EvalNotification -Action $action -Detail $detail
"END_NOTIFICATION=$endMail" | Add-Content -LiteralPath $statusLog -Encoding utf8
exit $exitCode
