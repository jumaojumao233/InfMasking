param(
    [int]$GpuIndex = 0,
    [int]$MaxEpochs = 10,
    [int]$MaxSize = 10000,
    [int]$PairSeed = 42,
    [string]$DataRoot = "C:/Users/liangyelian/work/InfMasking_desktop_20260923/dataset/data/trifeatures_g0_seed20260924",
    [switch]$EnableLinearProbe
)

$ErrorActionPreference = "Continue"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$startScript = Join-Path $PSScriptRoot "winpc_start_experiment.ps1"
$logRoot = Join-Path $root "winpc_logs"
$controllerLog = Join-Path $logRoot "winpc_g4_g0_sequence.status.log"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Write-ControllerLog {
    param([string]$Message)
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $controllerLog
}

function Read-RunExitCode {
    param([string]$RunName)
    $statusPath = Join-Path $logRoot "$RunName.status.log"
    if (-not (Test-Path -LiteralPath $statusPath)) {
        return $null
    }
    $endMatch = Get-Content -LiteralPath $statusPath -ErrorAction SilentlyContinue |
        Select-String -Pattern '^END .* EXIT_CODE=(-?\d+)$' |
        Select-Object -Last 1
    if ($null -eq $endMatch) {
        return $null
    }
    return [int]$endMatch.Matches[0].Groups[1].Value
}

function Wait-RunCompletion {
    param([string]$RunName)
    Write-ControllerLog "WAIT run=$RunName"
    while ($true) {
        $exitCode = Read-RunExitCode -RunName $RunName
        if ($null -ne $exitCode) {
            Write-ControllerLog "DONE run=$RunName EXIT_CODE=$exitCode"
            return $exitCode
        }
        Start-Sleep -Seconds 60
    }
}

function Start-G0Run {
    param(
        [string]$RunName,
        [ValidateSet("baseline", "unigir")]
        [string]$Method,
        [int]$Seed
    )

    $existingExit = Read-RunExitCode -RunName $RunName
    if ($null -ne $existingExit) {
        Write-ControllerLog "SKIP_EXISTING run=$RunName EXIT_CODE=$existingExit"
        return $existingExit
    }

    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $startScript,
        "-RunName", $RunName,
        "-Method", $Method,
        "-Seed", "$Seed",
        "-GpuIndex", "$GpuIndex",
        "-MaxEpochs", "$MaxEpochs",
        "-MaxSize", "$MaxSize",
        "-PairSeed", "$PairSeed",
        "-DataRoot", $DataRoot
    )
    if ($EnableLinearProbe) {
        $arguments += "-EnableLinearProbe"
    }

    Write-ControllerLog "START run=$RunName METHOD=$Method SEED=$Seed"
    & powershell.exe @arguments
    $processExit = $LASTEXITCODE
    $statusExit = Read-RunExitCode -RunName $RunName
    if ($null -eq $statusExit) {
        Write-ControllerLog "ERROR run=$RunName process_exit=$processExit status_exit=missing"
        return 1
    }
    Write-ControllerLog "RETURN run=$RunName process_exit=$processExit status_exit=$statusExit"
    return $statusExit
}

Set-Content -LiteralPath $controllerLog -Value "$(Get-Date -Format o) START G0 sequence"

$runs = @(
    [pscustomobject]@{ RunName = "winpc_g4_g0_trifeatures_s42_unigir"; Method = "unigir"; Seed = 42; Previous = "winpc_g4_g0_trifeatures_s42_baseline" },
    [pscustomobject]@{ RunName = "winpc_g4_g0_trifeatures_s7_baseline"; Method = "baseline"; Seed = 7; Previous = "winpc_g4_g0_trifeatures_s42_unigir" },
    [pscustomobject]@{ RunName = "winpc_g4_g0_trifeatures_s7_unigir"; Method = "unigir"; Seed = 7; Previous = "winpc_g4_g0_trifeatures_s7_baseline" },
    [pscustomobject]@{ RunName = "winpc_g4_g0_trifeatures_s123_baseline"; Method = "baseline"; Seed = 123; Previous = "winpc_g4_g0_trifeatures_s7_unigir" },
    [pscustomobject]@{ RunName = "winpc_g4_g0_trifeatures_s123_unigir"; Method = "unigir"; Seed = 123; Previous = "winpc_g4_g0_trifeatures_s123_baseline" }
)

foreach ($run in $runs) {
    $previousExit = Wait-RunCompletion -RunName $run.Previous
    if ($previousExit -ne 0) {
        Write-ControllerLog "STOP previous=$($run.Previous) EXIT_CODE=$previousExit next=$($run.RunName)"
        exit $previousExit
    }
    Start-Sleep -Seconds 15
    $currentExit = Start-G0Run -RunName $run.RunName -Method $run.Method -Seed $run.Seed
    if ($currentExit -ne 0) {
        Write-ControllerLog "STOP run=$($run.RunName) EXIT_CODE=$currentExit"
        exit $currentExit
    }
}

Write-ControllerLog "COMPLETE G0 sequence"
exit 0
