param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [string]$TaskName = "InfMasking-Experiment-Watchdog-liangyl",
    [ValidateRange(1, 1440)]
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Stop"

$watchdogScript = Join-Path $PSScriptRoot "winpc_experiment_watchdog.ps1"
$argument = "-NoProfile -ExecutionPolicy Bypass -File `"$watchdogScript`" -ConfigPath `"$ConfigPath`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$userId = (Get-CimInstance Win32_ComputerSystem).UserName
$principal = New-ScheduledTaskPrincipal `
    -UserId $userId `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Force | Out-Null

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $watchdogScript -ConfigPath $ConfigPath
Write-Output "Scheduled $TaskName every $IntervalMinutes minutes."
