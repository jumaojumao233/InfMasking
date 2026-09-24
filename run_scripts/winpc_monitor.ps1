param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
)

$ErrorActionPreference = "Continue"

$monitorRoot = Join-Path $ProjectRoot "winpc_monitor"
$monitorLog = Join-Path $monitorRoot "monitor.log"
$runRoot = Join-Path $ProjectRoot "winpc_runs"
$logRoot = Join-Path $ProjectRoot "winpc_logs"
New-Item -ItemType Directory -Force -Path $monitorRoot | Out-Null

$lines = @(
    "===== $(Get-Date -Format o) =====",
    "HOST=$env:COMPUTERNAME USER=$env:USERDOMAIN\$env:USERNAME"
)

$gpu = & nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>&1
$lines += "GPU:"
$lines += $gpu

$processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*$ProjectRoot*" }
if ($null -eq $processes) {
    $lines += "PROJECT_PYTHON=none"
}
else {
    $lines += "PROJECT_PYTHON:"
    foreach ($process in $processes) {
        $lines += "PID=$($process.ProcessId) COMMAND=$($process.CommandLine)"
    }
}

if (Test-Path -LiteralPath $runRoot) {
    $checkpoints = Get-ChildItem -LiteralPath $runRoot -Filter "last.ckpt" -File -Recurse
    if ($null -eq $checkpoints) {
        $lines += "CHECKPOINTS=none"
    }
    else {
        $lines += "CHECKPOINTS:"
        foreach ($checkpoint in $checkpoints) {
            $lines += "$($checkpoint.FullName) SIZE=$($checkpoint.Length) UPDATED=$($checkpoint.LastWriteTime.ToString('o'))"
        }
    }
}

if (Test-Path -LiteralPath $logRoot) {
    $statusLogs = Get-ChildItem -LiteralPath $logRoot -Filter "*.status.log" -File
    if ($null -ne $statusLogs) {
        $lines += "STATUS_LOGS:"
        foreach ($statusLog in $statusLogs) {
            $tail = (Get-Content -LiteralPath $statusLog.FullName -Tail 1 -ErrorAction SilentlyContinue) -join " "
            $lines += "$($statusLog.Name): $tail"
        }
    }
}

$lines | Add-Content -LiteralPath $monitorLog
