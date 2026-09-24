param(
    [string]$TaskName = "InfMasking-Winpc-G0Data-Seed20260924"
)

$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $PSScriptRoot "winpc_generate_g0_trifeatures.cmd"
$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/d /c `"$runScript`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
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

Write-Output "Scheduled $TaskName to start in about one minute."
