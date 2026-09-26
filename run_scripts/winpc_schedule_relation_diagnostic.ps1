param(
    [Parameter(Mandatory = $true)]
    [string]$TaskName,
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$DiagnosticScript = "run_scripts\diagnose_fusion_relation_structure.py",
    [int]$ModelSeed = 42,
    [int]$BatchSize = 64,
    [int]$MaxSamples = 256,
    [int]$TrainSamples = 512,
    [int]$TestSamples = 256,
    [double]$RidgeAlpha = 1.0,
    [string]$MaskRatios = "0 0.7 0.9",
    [string]$MaskSeeds = "101 202 303",
    [int]$GpuIndex = 0,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$startScript = Join-Path $PSScriptRoot "winpc_start_relation_diagnostic.ps1"
$argumentList = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File `"$startScript`"",
    "-RunName `"$RunName`"",
    "-OutputDir `"$OutputDir`"",
    "-Checkpoint `"$Checkpoint`"",
    "-DataRoot `"$DataRoot`"",
    "-DiagnosticScript `"$DiagnosticScript`"",
    "-ModelSeed $ModelSeed",
    "-BatchSize $BatchSize",
    "-MaxSamples $MaxSamples",
    "-TrainSamples $TrainSamples",
    "-TestSamples $TestSamples",
    "-RidgeAlpha $RidgeAlpha",
    "-GpuIndex $GpuIndex",
    "-Device `"$Device`""
)

if (-not [string]::IsNullOrWhiteSpace($MaskRatios)) {
    $argumentList += "-MaskRatios `"$MaskRatios`""
}
if (-not [string]::IsNullOrWhiteSpace($MaskSeeds)) {
    $argumentList += "-MaskSeeds `"$MaskSeeds`""
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
