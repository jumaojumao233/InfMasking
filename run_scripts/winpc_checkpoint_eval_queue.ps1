param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

function Resolve-QueuePath {
    param([string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) { return $Value }
    return (Join-Path $ProjectRoot $Value)
}

$config = Get-Content -LiteralPath (Resolve-QueuePath $ConfigPath) -Raw | ConvertFrom-Json
$experimentName = [string]$config.ExperimentName
$queueStatus = Join-Path $ProjectRoot "winpc_logs\${experimentName}.queue.status.log"
$queueError = Join-Path $ProjectRoot "winpc_logs\${experimentName}.queue.stderr.log"
$evalScript = Join-Path $ProjectRoot "run_scripts\winpc_checkpoint_eval.ps1"
New-Item -ItemType Directory -Force -Path (Split-Path $queueStatus) | Out-Null
"QUEUE_START $(Get-Date -Format o)" | Set-Content -LiteralPath $queueStatus -Encoding utf8

foreach ($run in @($config.Runs)) {
    $checkpoint = Resolve-QueuePath ([string]$run.CheckpointPath)
    $dataRoot = Resolve-QueuePath ([string]$run.DataRoot)
    $outputRoot = Resolve-QueuePath ([string]$run.OutputRoot)
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $evalScript,
        "-RunName", [string]$run.RunName,
        "-Method", [string]$run.Method,
        "-Seed", [string]$run.Seed,
        "-CheckpointPath", $checkpoint,
        "-DataRoot", $dataRoot,
        "-OutputRoot", $outputRoot,
        "-GpuIndex", [string]$run.GpuIndex,
        "-ProjectRoot", $ProjectRoot
    )
    if ([bool]$run.ProfileOnly) {
        $args += "-ProfileOnly"
    }
    "RUN_START $(Get-Date -Format o) $($run.RunName)" | Add-Content -LiteralPath $queueStatus -Encoding utf8
    try {
        & powershell.exe @args 2>> $queueError
        $exitCode = $LASTEXITCODE
    }
    catch {
        $_ | Out-String | Add-Content -LiteralPath $queueError -Encoding utf8
        $exitCode = 1
    }
    "RUN_END $(Get-Date -Format o) $($run.RunName) EXIT_CODE=$exitCode" | Add-Content -LiteralPath $queueStatus -Encoding utf8
    if ($exitCode -ne 0) {
        "QUEUE_STOP $(Get-Date -Format o) failed run $($run.RunName)" | Add-Content -LiteralPath $queueStatus -Encoding utf8
        exit $exitCode
    }
}

"QUEUE_COMPLETE $(Get-Date -Format o)" | Add-Content -LiteralPath $queueStatus -Encoding utf8
exit 0
