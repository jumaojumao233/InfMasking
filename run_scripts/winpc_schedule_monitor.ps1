param(
    [string]$TaskName = "InfMasking-Winpc-Monitor-liangyl"
)

$ErrorActionPreference = "Stop"

$monitorScript = Join-Path $PSScriptRoot "winpc_monitor.ps1"
$argument = "-NoProfile -ExecutionPolicy Bypass -File `"$monitorScript`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
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

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $monitorScript
Write-Output "Scheduled $TaskName every 10 minutes."
