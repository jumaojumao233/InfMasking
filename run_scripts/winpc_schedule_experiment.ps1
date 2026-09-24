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
    [int]$MaxSize = 1024,
    [int]$PairSeed = 42,
    [string]$DataRoot = "",
    [string]$ResumeCkptPath = "",
    [int]$QueueSize = 1024,
    [switch]$EnableLinearProbe,
    [switch]$ShuffleProfile,
    [switch]$CudaLaunchBlocking
)

$ErrorActionPreference = "Stop"

$startScript = Join-Path $PSScriptRoot "winpc_start_experiment.ps1"
$argumentList = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File `"$startScript`"",
    "-RunName `"$RunName`"",
    "-Method $Method",
    "-Seed $Seed",
    "-GpuIndex $GpuIndex",
    "-MaxEpochs $MaxEpochs",
    "-MaxSize $MaxSize",
    "-PairSeed $PairSeed",
    "-QueueSize $QueueSize"
)
if ($DataRoot -ne "") {
    $argumentList += "-DataRoot `"$DataRoot`""
}
if ($ResumeCkptPath -ne "") {
    $argumentList += "-ResumeCkptPath `"$ResumeCkptPath`""
}
if ($EnableLinearProbe) {
    $argumentList += "-EnableLinearProbe"
}
if ($ShuffleProfile) {
    $argumentList += "-ShuffleProfile"
}
if ($CudaLaunchBlocking) {
    $argumentList += "-CudaLaunchBlocking"
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
