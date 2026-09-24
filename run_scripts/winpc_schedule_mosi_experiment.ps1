param(
    [Parameter(Mandatory = $true)]
    [string]$TaskName,
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [ValidateSet("baseline", "unigir")]
    [string]$Method,
    [Parameter(Mandatory = $true)]
    [int]$Seed,
    [int]$GpuIndex = 0,
    [int]$MaxEpochs = 10,
    [int]$BatchSize = 32,
    [ValidateSet("by_epoch", "by_fit")]
    [string]$ProbeFrequency = "by_fit"
)

$ErrorActionPreference = "Stop"

$startScript = Join-Path $PSScriptRoot "winpc_mosi_unigir_smoke.ps1"
$argumentList = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File `"$startScript`"",
    "-RunName `"$RunName`"",
    "-Method $Method",
    "-Seed $Seed",
    "-GpuIndex $GpuIndex",
    "-MaxEpochs $MaxEpochs",
    "-BatchSize $BatchSize",
    "-ProbeFrequency $ProbeFrequency"
)

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($argumentList -join " ")
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
