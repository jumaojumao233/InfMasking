param(
    [int]$RunnerPid = 24244,
    [int[]]$InitialPythonPids = @(31764, 32488)
)

$ErrorActionPreference = "SilentlyContinue"
$projectRoot = Split-Path -Parent $PSScriptRoot
$logPath = Join-Path $projectRoot "overnight_logs\thermal_guard.log"
$pythonPaths = @(
    "C:\Users\breeze\AppData\Local\Programs\Python\Python313\python.exe",
    (Join-Path $projectRoot ".venv\Scripts\python.exe")
)
$runStart = Get-Date
$affinityMask = [IntPtr]255

function Get-TargetPython {
    Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        $pathMatches = $_.Path -in $pythonPaths
        $knownProcess = $_.Id -in $InitialPythonPids
        $newProcess = $false
        try { $newProcess = $_.StartTime -ge $runStart } catch {}
        $pathMatches -and ($knownProcess -or $newProcess)
    }
}

Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) thermal guard started; limiting target processes to 8 logical CPUs."

while ($true) {
    $targets = @(Get-TargetPython)
    foreach ($process in $targets) {
        try { $process.PriorityClass = "BelowNormal" } catch {}
        try { $process.ProcessorAffinity = $affinityMask } catch {}
    }

    $runner = Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue
    if ($null -eq $runner -and $targets.Count -eq 0) {
        Start-Sleep -Seconds 60
        $runnerCheck = Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue
        $targetCheck = @(Get-TargetPython)
        if ($null -eq $runnerCheck -and $targetCheck.Count -eq 0) {
            Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) overnight task finished; hibernating."
            & "$env:WINDIR\System32\shutdown.exe" /h
            break
        }
    }

    Start-Sleep -Seconds 15
}
