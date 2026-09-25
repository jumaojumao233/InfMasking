param(
    [string]$ProjectRoot = "",
    [string]$UserSuffix = "liangyl",
    [int]$GpuIndex = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

$logRoot = Join-Path $ProjectRoot "winpc_logs"
$runRoot = Join-Path $ProjectRoot "winpc_runs\InfMasking\bimodal_trifeatures"
$scheduleScript = Join-Path $ProjectRoot "run_scripts\winpc_schedule_experiment.ps1"
$dataRoot = Join-Path $ProjectRoot "dataset\data\trifeatures_g0_seed20260924"
$statePath = Join-Path $logRoot "winpc_g4_watchdog.state.json"
$statusPath = Join-Path $logRoot "winpc_g4_watchdog.status.json"
$eventPath = Join-Path $logRoot "winpc_g4_watchdog.events.log"
$userName = $env:USERNAME
$now = Get-Date

if ($DryRun) {
    $statePath = Join-Path $logRoot "winpc_g4_watchdog.dryrun.state.json"
    $statusPath = Join-Path $logRoot "winpc_g4_watchdog.dryrun.status.json"
    $eventPath = Join-Path $logRoot "winpc_g4_watchdog.dryrun.events.log"
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$runs = @(
    [pscustomobject]@{
        Order = 1
        TaskName = "InfMasking-G4-G0-s42-Baseline-$UserSuffix"
        RunName = "winpc_g4_g0_trifeatures_s42_baseline"
        Method = "baseline"
        Seed = 42
        QueueSize = 1024
    }
    [pscustomobject]@{
        Order = 2
        TaskName = "InfMasking-G4-G0-s42-UniGIR-$UserSuffix"
        RunName = "winpc_g4_g0_trifeatures_s42_unigir"
        Method = "unigir"
        Seed = 42
        QueueSize = 1024
    }
    [pscustomobject]@{
        Order = 3
        TaskName = "InfMasking-G4-G0-s7-Baseline-Retry1-$UserSuffix"
        RunName = "winpc_g4_g0_trifeatures_s7_baseline_retry1"
        Method = "baseline"
        Seed = 7
        QueueSize = 1024
    }
    [pscustomobject]@{
        Order = 4
        TaskName = "InfMasking-G4-G0-s7-UniGIR-$UserSuffix"
        RunName = "winpc_g4_g0_trifeatures_s7_unigir"
        Method = "unigir"
        Seed = 7
        QueueSize = 1024
    }
    [pscustomobject]@{
        Order = 5
        TaskName = "InfMasking-G4-G0-s123-Baseline-$UserSuffix"
        RunName = "winpc_g4_g0_trifeatures_s123_baseline"
        Method = "baseline"
        Seed = 123
        QueueSize = 1024
    }
    [pscustomobject]@{
        Order = 6
        TaskName = "InfMasking-G4-G0-s123-UniGIR-$UserSuffix"
        RunName = "winpc_g4_g0_trifeatures_s123_unigir"
        Method = "unigir"
        Seed = 123
        QueueSize = 1024
    }
)

function Get-ErrorMarker {
    param([string[]]$Paths)

    $patterns = @(
        "CUDA error",
        "CUDA out of memory",
        "Traceback (most recent call last)",
        "RuntimeError:",
        "OutOfMemoryError",
        "Exception:"
    )

    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath $path)) {
            continue
        }
        foreach ($pattern in $patterns) {
            if (Select-String -LiteralPath $path -SimpleMatch -Pattern $pattern -Quiet) {
                return $pattern
            }
        }
    }
    return ""
}

function Get-TaskSnapshot {
    param([object]$Run)

    $taskState = "Missing"
    $lastRun = ""
    $lastResult = ""
    $nextRun = ""
    $taskExists = $false

    try {
        $task = Get-ScheduledTask -TaskName $Run.TaskName -ErrorAction Stop
        $taskInfo = Get-ScheduledTaskInfo -TaskName $Run.TaskName -TaskPath $task.TaskPath -ErrorAction Stop
        $taskState = [string]$task.State
        $lastRun = [string]$taskInfo.LastRunTime
        $lastResult = [string]$taskInfo.LastTaskResult
        $nextRun = [string]$taskInfo.NextRunTime
        $taskExists = $true
    }
    catch {
        $taskExists = $false
    }

    $statusLog = Join-Path $logRoot "$($Run.RunName).status.log"
    $stdoutLog = Join-Path $logRoot "$($Run.RunName).stdout.log"
    $stderrLog = Join-Path $logRoot "$($Run.RunName).stderr.log"
    $checkpoint = Join-Path $runRoot "$($Run.RunName)\checkpoints\last.ckpt"
    $statusText = ""
    if (Test-Path -LiteralPath $statusLog) {
        $statusText = Get-Content -LiteralPath $statusLog -Raw
    }

    $errorMarker = Get-ErrorMarker -Paths @($statusLog, $stdoutLog, $stderrLog)
    $endMatch = [regex]::Match($statusText, "END\s+\S+\s+EXIT_CODE=(\d+)")
    $exitCode = ""
    $state = "Missing"
    $detail = "status log missing"

    if ($errorMarker -ne "") {
        $state = "Failed"
        $detail = "error marker: $errorMarker"
    }
    elseif ($endMatch.Success) {
        $exitCode = $endMatch.Groups[1].Value
        if ($exitCode -ne "0") {
            $state = "Failed"
            $detail = "EXIT_CODE=$exitCode"
        }
        elseif (-not (Test-Path -LiteralPath $checkpoint)) {
            $state = "CheckpointMissing"
            $detail = "successful end without last.ckpt"
        }
        else {
            $state = "Succeeded"
            $detail = "EXIT_CODE=0 and last.ckpt exists"
        }
    }
    elseif ($taskState -eq "Running") {
        $state = "Running"
        $detail = "scheduled task is running"
    }
    elseif ($taskExists) {
        $state = "Pending"
        $detail = "scheduled task exists without END"
    }

    [pscustomobject]@{
        Order = $Run.Order
        TaskName = $Run.TaskName
        RunName = $Run.RunName
        Method = $Run.Method
        Seed = $Run.Seed
        State = $state
        Detail = $detail
        TaskExists = $taskExists
        TaskState = $taskState
        LastRun = $lastRun
        LastResult = $lastResult
        NextRun = $nextRun
        ExitCode = $exitCode
        StatusLog = $statusLog
        Checkpoint = $checkpoint
        CheckpointExists = (Test-Path -LiteralPath $checkpoint)
    }
}

function Get-GpuSnapshot {
    $apps = @()
    try {
        $raw = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)
        if ($LASTEXITCODE -ne 0) {
            return [pscustomobject]@{
                Safe = $false
                Detail = "nvidia-smi failed with exit code $LASTEXITCODE"
                Apps = @()
            }
        }

        foreach ($line in $raw) {
            $text = ([string]$line).Trim()
            if ($text -eq "" -or $text -like "No running processes found*") {
                continue
            }
            $parts = $text -split ",\s*"
            $processId = 0
            if (-not [int]::TryParse($parts[0], [ref]$processId)) {
                return [pscustomobject]@{
                    Safe = $false
                    Detail = "cannot parse nvidia-smi row: $text"
                    Apps = @()
                }
            }

            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
            $ownerName = ""
            $commandLine = ""
            if ($null -ne $process) {
                $commandLine = [string]$process.CommandLine
                $owner = Invoke-CimMethod -InputObject $process -MethodName GetOwner -ErrorAction SilentlyContinue
                if ($null -ne $owner) {
                    $ownerName = [string]$owner.User
                }
            }

            $knownRun = $false
            foreach ($run in $runs) {
                if ($commandLine -like "*$($run.RunName)*") {
                    $knownRun = $true
                    break
                }
            }

            $processPath = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
            $processName = $processPath
            if ($processName -eq "[Insufficient Permissions]") {
                $fallbackProcess = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($null -ne $fallbackProcess) {
                    $processName = [string]$fallbackProcess.ProcessName
                    $processPath = $processName
                }
            }
            $desktopProcess = ($processPath + " " + $processName + " " + $commandLine) -match "(?i)\\Windows\\|\\WindowsApps\\microsoft\.|\\Microsoft\\|\\Edge\\|dwm\.exe|WUDFHost\.exe|^dwm$|^WUDFHost$|^csrss$|^winlogon$|^explorer$|^SearchApp$|^ShellExperienceHost$|^TextInputHost$|^msedgewebview2$|^msedge$|^PhoneExperienceHost$"
            $allowed = $desktopProcess -or ($ownerName -eq $userName -and $knownRun)
            $apps += [pscustomobject]@{
                Pid = $processId
                ProcessName = $processName
                UsedMemory = if ($parts.Count -gt 2) { $parts[2].Trim() } else { "" }
                Owner = $ownerName
                CommandLine = $commandLine
                Allowed = $allowed
                DesktopProcess = $desktopProcess
            }
        }
    }
    catch {
        return [pscustomobject]@{
            Safe = $false
            Detail = "GPU inspection failed: $($_.Exception.Message)"
            Apps = @()
        }
    }

    $unknown = @($apps | Where-Object { -not $_.Allowed })
    if ($unknown.Count -gt 0) {
        $description = ($unknown | ForEach-Object { "pid=$($_.Pid),owner=$($_.Owner),process=$($_.ProcessName)" }) -join "; "
        return [pscustomobject]@{
            Safe = $false
            Detail = "unknown GPU compute process: $description"
            Apps = $apps
        }
    }

    return [pscustomobject]@{
        Safe = $true
        Detail = if ($apps.Count -eq 0) { "no GPU compute process" } else { "only current G0 task process is using the GPU" }
        Apps = $apps
    }
}

function Start-NextRun {
    param([object]$Run)

    $cli = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $scheduleScript,
        "-TaskName", $Run.TaskName,
        "-RunName", $Run.RunName,
        "-Method", $Run.Method,
        "-Seed", [string]$Run.Seed,
        "-GpuIndex", [string]$GpuIndex,
        "-MaxEpochs", "10",
        "-MaxSize", "10000",
        "-PairSeed", "42",
        "-DataRoot", $dataRoot,
        "-EnableLinearProbe"
    )

    if ($Run.Method -eq "unigir") {
        $cli += @("-QueueSize", [string]$Run.QueueSize)
    }

    if ($DryRun) {
        return "DRY_RUN would schedule $($Run.TaskName)"
    }

    & powershell.exe @cli | Out-File -FilePath $eventPath -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "schedule script failed with exit code $LASTEXITCODE"
    }
    return "START_REQUESTED $($Run.TaskName)"
}

try {
    $snapshots = @($runs | ForEach-Object { Get-TaskSnapshot -Run $_ })
    $gpu = Get-GpuSnapshot
    $stopReasons = @()

    foreach ($snapshot in $snapshots) {
        if ($snapshot.State -in @("Failed", "CheckpointMissing")) {
            $stopReasons += "$($snapshot.RunName): $($snapshot.Detail)"
        }
    }
    if (-not $gpu.Safe) {
        $stopReasons += "GPU: $($gpu.Detail)"
    }

    $action = "WAIT"
    $actionDetail = ""
    if ($stopReasons.Count -gt 0) {
        $action = "STOP"
        $actionDetail = $stopReasons -join " | "
    }
    else {
        $active = @($snapshots | Where-Object { $_.State -in @("Running", "Pending") })
        $next = $snapshots | Where-Object { $_.State -ne "Succeeded" } | Select-Object -First 1
        $laterRuns = @()
        if ($null -ne $next) {
            $laterRuns = @($snapshots | Where-Object {
                $_.Order -gt $next.Order -and $_.State -ne "Missing"
            })
        }
        if ($laterRuns.Count -gt 0) {
            $action = "STOP"
            $actionDetail = "sequence violation after $($next.RunName): " + (($laterRuns | ForEach-Object { "$($_.RunName)=$($_.State)" }) -join ", ")
            $stopReasons += $actionDetail
        }
        elseif ($null -eq $next) {
            $action = "ALL_COMPLETE"
            $actionDetail = "all six G0 tasks succeeded and have checkpoints"
        }
        elseif ($active.Count -gt 0) {
            $action = "WAIT"
            $actionDetail = "active or pending task: " + (($active | ForEach-Object { $_.RunName }) -join ", ")
        }
        elseif ($next.State -eq "Missing") {
            $action = "START_REQUESTED"
            $actionDetail = Start-NextRun -Run $runs[$next.Order - 1]
        }
        else {
            $action = "WAIT"
            $actionDetail = "next task is in state $($next.State): $($next.Detail)"
        }
    }

    $state = $null
    if (Test-Path -LiteralPath $statePath) {
        try { $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } catch { $state = $null }
    }
    if ($null -eq $state -or $state.LastAction -ne $action -or $action -in @("STOP", "START_REQUESTED", "ALL_COMPLETE")) {
        "$(Get-Date -Format o) ACTION=$action DETAIL=$actionDetail" | Add-Content -LiteralPath $eventPath -Encoding utf8
    }

    $result = [pscustomobject]@{
        Timestamp = (Get-Date -Format o)
        Host = $env:COMPUTERNAME
        User = $userName
        Action = $action
        Detail = $actionDetail
        StopReasons = $stopReasons
        Gpu = $gpu
        Runs = $snapshots
    }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
    [pscustomobject]@{
        LastAction = $action
        LastDetail = $actionDetail
        UpdatedAt = (Get-Date -Format o)
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
    Write-Output $result
}
catch {
    $detail = "watchdog exception: $($_.Exception.Message)"
    "$(Get-Date -Format o) ACTION=STOP DETAIL=$detail" | Add-Content -LiteralPath $eventPath -Encoding utf8
    [pscustomobject]@{
        Timestamp = (Get-Date -Format o)
        Host = $env:COMPUTERNAME
        User = $userName
        Action = "STOP"
        Detail = $detail
        StopReasons = @($detail)
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
    exit 0
}
