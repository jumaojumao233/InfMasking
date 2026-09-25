<#
    Run one read-only per-sample representation/profile diagnostic on winpc.
    This script never calls trainer.fit and never modifies a checkpoint.
#>

param(
    [Parameter(Mandatory = $true)] [string]$RunName,
    [Parameter(Mandatory = $true)] [ValidateSet("baseline", "unigir")] [string]$Method,
    [Parameter(Mandatory = $true)] [int]$Seed,
    [Parameter(Mandatory = $true)] [string]$CheckpointPath,
    [Parameter(Mandatory = $true)] [string]$DataRoot,
    [Parameter(Mandatory = $true)] [string]$OutputRoot,
    [int]$GpuIndex = 0,
    [int]$NotifyIntervalSeconds = 900,
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
$script = Join-Path $ProjectRoot "run_scripts\export_profile_representation_diagnostics.py"
$notifyScript = Join-Path $ProjectRoot "run_scripts\winpc_g4_notify.py"
$mailEnvScript = Join-Path $env:USERPROFILE "secrets\InfMasking\mail.env.ps1"
$logRoot = Join-Path $ProjectRoot "winpc_logs"
$stdoutLog = Join-Path $logRoot "$RunName.repdiag.stdout.log"
$stderrLog = Join-Path $logRoot "$RunName.repdiag.stderr.log"
$statusLog = Join-Path $logRoot "$RunName.repdiag.status.log"
$bodyPath = Join-Path $logRoot "$RunName.repdiag.email.body.txt"
$mailErrorPath = Join-Path $logRoot "$RunName.repdiag.mail.stderr.log"

New-Item -ItemType Directory -Force -Path $logRoot, $OutputRoot | Out-Null

function Send-DiagnosticNotification {
    param([string]$Action, [string]$Detail)
    if (-not (Test-Path -LiteralPath $mailEnvScript)) { return "MAIL_CONFIG_MISSING" }
    if (-not (Test-Path -LiteralPath $notifyScript)) { return "MAIL_SCRIPT_MISSING" }
    if (-not (Test-Path -LiteralPath $python)) { return "MAIL_PYTHON_MISSING" }
    try {
        . $mailEnvScript
        @(
            "Timestamp: $(Get-Date -Format o)",
            "Host: $env:COMPUTERNAME",
            "Diagnostic run: $RunName",
            "Method: $Method",
            "Seed: $Seed",
            "Action: $Action",
            "Detail: $Detail",
            "Checkpoint: $CheckpointPath",
            "Output directory: $OutputRoot"
        ) | Set-Content -LiteralPath $bodyPath -Encoding utf8
        & $python $notifyScript --subject "[InfMasking representation diagnostic][$Action] $RunName" --body-file $bodyPath 2>> $mailErrorPath | Out-Null
        if ($LASTEXITCODE -eq 0) { return "EMAIL_SENT" }
        return "EMAIL_FAILED_EXIT_$LASTEXITCODE"
    }
    catch { return "EMAIL_FAILED: $($_.Exception.Message)" }
}

if (-not (Test-Path -LiteralPath $CheckpointPath)) {
    "$(Get-Date -Format o) ACTION=STOP DETAIL=checkpoint missing: $CheckpointPath" | Set-Content -LiteralPath $statusLog -Encoding utf8
    Send-DiagnosticNotification -Action "STOP" -Detail "checkpoint missing" | Out-Null
    exit 2
}

$args = @(
    $script,
    "--checkpoint", $CheckpointPath,
    "--data-root", $DataRoot,
    "--output-dir", $OutputRoot,
    "--method", $Method,
    "--seed", [string]$Seed,
    "--device", "cuda:0",
    "--project-root", $ProjectRoot
)
$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
"START $(Get-Date -Format o)" | Set-Content -LiteralPath $statusLog -Encoding utf8
"METHOD=$Method SEED=$Seed CHECKPOINT=$CheckpointPath DATA_ROOT=$DataRoot OUTPUT_ROOT=$OutputRoot GPU_INDEX=$GpuIndex" | Add-Content -LiteralPath $statusLog
$startMail = Send-DiagnosticNotification -Action "START" -Detail "read-only diagnostic requested; no training will run"
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
                $mail = Send-DiagnosticNotification -Action "WAIT" -Detail "diagnostic still running; process id=$($process.Id)"
                "WAIT $(Get-Date -Format o) PID=$($process.Id) NOTIFICATION=$mail" | Add-Content -LiteralPath $statusLog -Encoding utf8
                $lastMailAt = Get-Date
            }
        }
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = [int]$process.ExitCode
    $stderrText = if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Raw } else { "" }
    if ($exitCode -eq 0 -and $stderrText -match "Traceback \(most recent call last\)|CUDA error|Exception:") {
        $exitCode = 1
    }
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrLog -Encoding utf8
    $exitCode = 1
}
finally { Pop-Location }

"END $(Get-Date -Format o) EXIT_CODE=$exitCode" | Add-Content -LiteralPath $statusLog -Encoding utf8
$action = if ($exitCode -eq 0) { "COMPLETE" } else { "FAILED" }
$endMail = Send-DiagnosticNotification -Action $action -Detail "diagnostic process exit code $exitCode"
"END_NOTIFICATION=$endMail" | Add-Content -LiteralPath $statusLog -Encoding utf8
exit $exitCode
