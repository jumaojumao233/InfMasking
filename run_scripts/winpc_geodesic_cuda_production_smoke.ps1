param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RunName = "winpc_geodesic_cuda_production_smoke_s7"
)

$ErrorActionPreference = "Continue"
$python = Join-Path $Root ".venv\Scripts\python.exe"
$diagnostic = Join-Path $Root "run_scripts\geodesic_cuda_production_smoke.py"
$logRoot = Join-Path $Root "winpc_cuda_diag_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$stdoutLog = Join-Path $logRoot "$RunName.stdout.log"
$stderrLog = Join-Path $logRoot "$RunName.stderr.log"
$statusLog = Join-Path $logRoot "$RunName.status.log"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_LAUNCH_BLOCKING = "1"
"START $(Get-Date -Format o)" | Set-Content -LiteralPath $statusLog
"PYTHON=$python" | Add-Content -LiteralPath $statusLog
"DIAGNOSTIC=$diagnostic" | Add-Content -LiteralPath $statusLog
"CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=$env:CUDA_LAUNCH_BLOCKING" | Add-Content -LiteralPath $statusLog
"STDOUT=$stdoutLog" | Add-Content -LiteralPath $statusLog
"STDERR=$stderrLog" | Add-Content -LiteralPath $statusLog

$exitCode = 1
Push-Location $Root
try {
    & $python $diagnostic 1> $stdoutLog 2> $stderrLog
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
