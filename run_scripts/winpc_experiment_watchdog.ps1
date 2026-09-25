<#
    Configuration-driven watchdog for detached winpc experiments.

    The script only inspects tasks, logs, checkpoints and GPU processes. It
    schedules the next configured run only when all safety checks pass.
    Configure the Windows Task Scheduler to run it every 15 minutes.
#>

param(
    [string]$ProjectRoot = "",
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

function Resolve-ConfiguredPath {
    param([string]$PathValue)

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($PathValue)
    $expanded = $expanded.Replace("`$HOME", $env:USERPROFILE)
    $expanded = $expanded.Replace("~", $env:USERPROFILE)
    if ([IO.Path]::IsPathRooted($expanded)) {
        return $expanded
    }
    return (Join-Path $ProjectRoot $expanded)
}

$configFullPath = Resolve-ConfiguredPath -PathValue $ConfigPath
if (-not (Test-Path -LiteralPath $configFullPath)) {
    throw "watchdog config does not exist: $configFullPath"
}
$config = Get-Content -LiteralPath $configFullPath -Raw | ConvertFrom-Json

$experimentName = [string]$config.ExperimentName
if ([string]::IsNullOrWhiteSpace($experimentName)) {
    throw "ExperimentName is required in watchdog config"
}
$safeName = ($experimentName -replace "[^A-Za-z0-9_.-]", "_")
$userName = if ($config.UserSuffix) { [string]$config.UserSuffix } else { $env:USERNAME }
$gpuIndex = if ($null -ne $config.GpuIndex) { [int]$config.GpuIndex } else { 0 }
$logRoot = Resolve-ConfiguredPath -PathValue ([string]$config.LogRoot)
$runRoot = Resolve-ConfiguredPath -PathValue ([string]$config.RunRoot)
$scheduleScript = Resolve-ConfiguredPath -PathValue ([string]$config.ScheduleScript)
$notifyScript = Resolve-ConfiguredPath -PathValue ([string]$config.NotifyScript)
$notifyPython = Resolve-ConfiguredPath -PathValue ([string]$config.NotifyPython)
$mailEnvScript = Resolve-ConfiguredPath -PathValue ([string]$config.MailEnvScript)
$defaultDataRoot = Resolve-ConfiguredPath -PathValue ([string]$config.DataRoot)
$statePath = Join-Path $logRoot "$safeName.watchdog.state.json"
$statusPath = Join-Path $logRoot "$safeName.watchdog.status.json"
$eventPath = Join-Path $logRoot "$safeName.watchdog.events.log"

if ($DryRun) {
    $statePath = Join-Path $logRoot "$safeName.watchdog.dryrun.state.json"
    $statusPath = Join-Path $logRoot "$safeName.watchdog.dryrun.status.json"
    $eventPath = Join-Path $logRoot "$safeName.watchdog.dryrun.events.log"
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$runs = @($config.Runs | ForEach-Object {
    if ($null -eq $_.Order -or $null -eq $_.TaskName -or $null -eq $_.RunName -or $null -eq $_.Method -or $null -eq $_.Seed) {
        throw "every configured run needs Order, TaskName, RunName, Method and Seed"
    }
    $_
} | Sort-Object { [int]$_.Order })
if ($runs.Count -eq 0) {
    throw "watchdog config must contain at least one run"
}

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
    if ($config.ErrorMarkers) {
        $patterns = @($config.ErrorMarkers | ForEach-Object { [string]$_ })
    }

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

function Get-RunProcess {
    param([string]$RunName)

    try {
        return @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction Stop |
            Where-Object { [string]$_.CommandLine -like "*$RunName*" })
    }
    catch {
        return @()
    }
}

function Get-TaskSnapshot {
    param([object]$Run)

    $taskState = "Missing"
    $lastRun = ""
    $lastResult = ""
    $nextRun = ""
    $taskExists = $false
    try {
        $task = Get-ScheduledTask -TaskName ([string]$Run.TaskName) -ErrorAction Stop
        $taskInfo = Get-ScheduledTaskInfo -TaskName ([string]$Run.TaskName) -TaskPath $task.TaskPath -ErrorAction Stop
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
    $checkpoint = if ($Run.CheckpointPath) {
        Resolve-ConfiguredPath -PathValue ([string]$Run.CheckpointPath)
    }
    else {
        Join-Path $runRoot "$($Run.RunName)\checkpoints\last.ckpt"
    }
    $statusText = if (Test-Path -LiteralPath $statusLog) { Get-Content -LiteralPath $statusLog -Raw } else { "" }
    $errorMarker = Get-ErrorMarker -Paths @($statusLog, $stdoutLog, $stderrLog)
    $endMatch = [regex]::Match($statusText, "END\s+\S+\s+EXIT_CODE=(\d+)")
    $exitCode = ""
    $state = "Missing"
    $detail = "status log missing"
    $processes = @(Get-RunProcess -RunName ([string]$Run.RunName))

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
    elseif ($processes.Count -gt 0 -or $taskState -eq "Running") {
        $state = "Running"
        $detail = "matching training process or scheduled task is running"
    }
    elseif ($taskExists) {
        $state = "Pending"
        $detail = "scheduled task exists without END"
    }

    [pscustomobject]@{
        Order = [int]$Run.Order
        TaskName = [string]$Run.TaskName
        RunName = [string]$Run.RunName
        Method = [string]$Run.Method
        Seed = [int]$Run.Seed
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
            return [pscustomobject]@{ Safe = $false; Detail = "nvidia-smi failed with exit code $LASTEXITCODE"; Apps = @() }
        }
        foreach ($line in $raw) {
            $text = ([string]$line).Trim()
            if ($text -eq "" -or $text -like "No running processes found*") { continue }
            $parts = $text -split ",\s*"
            $processId = 0
            if (-not [int]::TryParse($parts[0], [ref]$processId)) {
                return [pscustomobject]@{ Safe = $false; Detail = "cannot parse nvidia-smi row: $text"; Apps = @() }
            }
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
            $ownerName = ""
            $commandLine = ""
            if ($null -ne $process) {
                $commandLine = [string]$process.CommandLine
                $owner = Invoke-CimMethod -InputObject $process -MethodName GetOwner -ErrorAction SilentlyContinue
                if ($null -ne $owner) { $ownerName = [string]$owner.User }
            }
            $knownRun = $false
            foreach ($run in $runs) {
                if ($commandLine -like "*$($run.RunName)*") { $knownRun = $true; break }
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
            $desktopProcess = (
                ($processPath -match "(?i)\\Windows\\|\\WindowsApps\\|\\Microsoft\\|\\Edge\\") -or
                ($commandLine -match "(?i)\\Windows\\|\\WindowsApps\\|\\Microsoft\\|\\Edge\\") -or
                ($processName -match "(?i)^(dwm|dwm\.exe|WUDFHost|WUDFHost\.exe|csrss|csrss\.exe|winlogon|winlogon\.exe|explorer|explorer\.exe|SearchApp|SearchApp\.exe|ShellExperienceHost|ShellExperienceHost\.exe|TextInputHost|TextInputHost\.exe|msedgewebview2|msedgewebview2\.exe|msedge|msedge\.exe|PhoneExperienceHost|PhoneExperienceHost\.exe)$")
            )
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
        return [pscustomobject]@{ Safe = $false; Detail = "GPU inspection failed: $($_.Exception.Message)"; Apps = @() }
    }
    $unknown = @($apps | Where-Object { -not $_.Allowed })
    if ($unknown.Count -gt 0) {
        $description = ($unknown | ForEach-Object { "pid=$($_.Pid),owner=$($_.Owner),process=$($_.ProcessName)" }) -join "; "
        return [pscustomobject]@{ Safe = $false; Detail = "unknown GPU compute process: $description"; Apps = $apps }
    }
    return [pscustomobject]@{
        Safe = $true
        Detail = if ($apps.Count -eq 0) { "no GPU compute process" } else { "only configured experiment or desktop GPU processes are present" }
        Apps = $apps
    }
}

function Add-OptionalSwitch {
    param([System.Collections.Generic.List[string]]$Arguments, [object]$Run, [string]$PropertyName, [string]$SwitchName)
    if ($Run.$PropertyName -eq $true) { $Arguments.Add($SwitchName) }
}

function Start-NextRun {
    param([object]$Run)
    if ($config.AutoStart -eq $false) { return "AUTO_START_DISABLED for $($Run.TaskName)" }
    $maxEpochs = if ($Run.MaxEpochs) { [string]$Run.MaxEpochs } else { "10" }
    $maxSize = if ($Run.MaxSize) { [string]$Run.MaxSize } else { "10000" }
    $pairSeed = if ($Run.PairSeed) { [string]$Run.PairSeed } else { "42" }
    $queueSize = if ($Run.QueueSize) { [string]$Run.QueueSize } else { "1024" }
    $cli = [System.Collections.Generic.List[string]]::new()
    $baseArguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scheduleScript,
        "-TaskName", [string]$Run.TaskName, "-RunName", [string]$Run.RunName,
        "-Method", [string]$Run.Method, "-Seed", [string]$Run.Seed,
        "-GpuIndex", [string]$gpuIndex,
        "-MaxEpochs", $maxEpochs,
        "-MaxSize", $maxSize,
        "-PairSeed", $pairSeed,
        "-QueueSize", $queueSize)
    foreach ($item in $baseArguments) {
        $cli.Add($item)
    }
    $dataRoot = if ($Run.DataRoot) { Resolve-ConfiguredPath -PathValue ([string]$Run.DataRoot) } else { $defaultDataRoot }
    if ($dataRoot -ne "") { $cli.Add("-DataRoot"); $cli.Add($dataRoot) }
    if ($Run.ResumeCkptPath) { $cli.Add("-ResumeCkptPath"); $cli.Add((Resolve-ConfiguredPath -PathValue ([string]$Run.ResumeCkptPath))) }
    Add-OptionalSwitch -Arguments $cli -Run $Run -PropertyName "EnableLinearProbe" -SwitchName "-EnableLinearProbe"
    Add-OptionalSwitch -Arguments $cli -Run $Run -PropertyName "ShuffleProfile" -SwitchName "-ShuffleProfile"
    Add-OptionalSwitch -Arguments $cli -Run $Run -PropertyName "CudaLaunchBlocking" -SwitchName "-CudaLaunchBlocking"

    if ($DryRun) { return "DRY_RUN would schedule $($Run.TaskName)" }
    & powershell.exe @cli | Out-File -FilePath $eventPath -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "schedule script failed with exit code $LASTEXITCODE" }
    return "START_REQUESTED $($Run.TaskName)"
}

function Send-WatchdogNotification {
    param([object]$Result)
    if ($DryRun) { return "DRY_RUN_SKIPPED" }
    foreach ($path in @($mailEnvScript, $notifyScript, $notifyPython)) {
        if (-not (Test-Path -LiteralPath $path)) { return "MAIL_DEPENDENCY_MISSING: $path" }
    }
    try {
        . $mailEnvScript
        $subject = "[InfMasking $experimentName][$($Result.Action)] $($Result.Detail)"
        $bodyLines = @(
            "Timestamp: $($Result.Timestamp)",
            "Host: $($Result.Host)",
            "Experiment: $experimentName",
            "Action: $($Result.Action)",
            "Detail: $($Result.Detail)",
            "",
            "Run states:"
        )
        foreach ($run in $Result.Runs) {
            $bodyLines += ("{0}. {1} seed={2} state={3} detail={4}" -f $run.Order, $run.Method, $run.Seed, $run.State, $run.Detail)
        }
        if ($Result.StopReasons.Count -gt 0) {
            $bodyLines += ""
            $bodyLines += "Stop reasons:"
            $bodyLines += $Result.StopReasons
        }
        $bodyPath = Join-Path $logRoot "$safeName.watchdog.email.body.txt"
        $bodyLines | Set-Content -LiteralPath $bodyPath -Encoding utf8
        $mailErrorPath = Join-Path $logRoot "$safeName.watchdog.mail.stderr.log"
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $notifyPython $notifyScript --subject $subject --body-file $bodyPath 2>> $mailErrorPath | Out-Null
            $mailExitCode = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $previousErrorActionPreference }
        if ($mailExitCode -eq 0) { return "EMAIL_SENT" }
        return "EMAIL_FAILED_EXIT_$mailExitCode"
    }
    catch { return "EMAIL_FAILED: $($_.Exception.Message)" }
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
    if (-not $gpu.Safe) { $stopReasons += "GPU: $($gpu.Detail)" }

    $action = "WAIT"
    $actionDetail = ""
    if ($stopReasons.Count -gt 0) {
        $action = "STOP"
        $actionDetail = $stopReasons -join " | "
    }
    else {
        $next = $snapshots | Where-Object { $_.State -ne "Succeeded" } | Select-Object -First 1
        $laterRuns = @()
        if ($null -ne $next) {
            $laterRuns = @($snapshots | Where-Object { $_.Order -gt $next.Order -and $_.State -ne "Missing" })
        }
        if ($laterRuns.Count -gt 0) {
            $action = "STOP"
            $actionDetail = "sequence violation after $($next.RunName): " + (($laterRuns | ForEach-Object { "$($_.RunName)=$($_.State)" }) -join ", ")
            $stopReasons += $actionDetail
        }
        elseif ($null -eq $next) {
            $action = "ALL_COMPLETE"
            $actionDetail = "all configured runs succeeded and have checkpoints"
        }
        elseif (@($snapshots | Where-Object { $_.State -in @("Running", "Pending") }).Count -gt 0) {
            $action = "WAIT"
            $actionDetail = "active or pending task: " + ((@($snapshots | Where-Object { $_.State -in @("Running", "Pending") }) | ForEach-Object { $_.RunName }) -join ", ")
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
    $actionChanged = ($null -eq $state -or $state.LastAction -ne $action)
    if ($null -eq $state -or $state.LastAction -ne $action -or $action -in @("STOP", "START_REQUESTED")) {
        "$(Get-Date -Format o) ACTION=$action DETAIL=$actionDetail" | Add-Content -LiteralPath $eventPath -Encoding utf8
    }

    $result = [pscustomobject]@{
        Timestamp = (Get-Date -Format o)
        Host = $env:COMPUTERNAME
        User = $userName
        Experiment = $experimentName
        Action = $action
        Detail = $actionDetail
        StopReasons = $stopReasons
        Gpu = $gpu
        Runs = $snapshots
    }
    $shouldNotify = ($action -eq "WAIT" -or $actionChanged -or $action -in @("STOP", "START_REQUESTED"))
    $notification = if ($shouldNotify) { Send-WatchdogNotification -Result $result } else { "SKIPPED" }
    $result | Add-Member -NotePropertyName Notification -NotePropertyValue $notification
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
    [pscustomobject]@{ LastAction = $action; LastDetail = $actionDetail; UpdatedAt = (Get-Date -Format o) } |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
    Write-Output $result
}
catch {
    $detail = "watchdog exception: $($_.Exception.Message)"
    "$(Get-Date -Format o) ACTION=STOP DETAIL=$detail" | Add-Content -LiteralPath $eventPath -Encoding utf8
    [pscustomobject]@{ Timestamp = (Get-Date -Format o); Host = $env:COMPUTERNAME; User = $userName; Experiment = $experimentName; Action = "STOP"; Detail = $detail; StopReasons = @($detail) } |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
    exit 0
}
