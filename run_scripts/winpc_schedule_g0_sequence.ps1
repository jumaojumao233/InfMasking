param(
    [Parameter(Mandatory = $true)]
    [string]$TaskName,
    [int]$GpuIndex = 0,
    [int]$MaxEpochs = 10,
    [int]$MaxSize = 10000,
    [int]$PairSeed = 42,
    [string]$DataRoot = "C:/Users/liangyelian/work/InfMasking_desktop_20260923/dataset/data/trifeatures_g0_seed20260924",
    [switch]$EnableLinearProbe
)

$ErrorActionPreference = "Stop"

$sequenceScript = Join-Path $PSScriptRoot "winpc_run_g0_sequence.ps1"
$argumentList = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File `"$sequenceScript`"",
    "-GpuIndex $GpuIndex",
    "-MaxEpochs $MaxEpochs",
    "-MaxSize $MaxSize",
    "-PairSeed $PairSeed",
    "-DataRoot `"$DataRoot`""
)
if ($EnableLinearProbe) {
    $argumentList += "-EnableLinearProbe"
}

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
