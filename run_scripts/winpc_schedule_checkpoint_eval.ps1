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
    [Parameter(Mandatory = $true)]
    [string]$CheckpointPath,
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [int]$GpuIndex = 0,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

$evalScript = Join-Path $ProjectRoot "run_scripts\winpc_checkpoint_eval.ps1"
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File `"$evalScript`"",
    "-RunName `"$RunName`"",
    "-Method $Method",
    "-Seed $Seed",
    "-CheckpointPath `"$CheckpointPath`"",
    "-DataRoot `"$DataRoot`"",
    "-OutputRoot `"$OutputRoot`"",
    "-GpuIndex $GpuIndex",
    "-ProjectRoot `"$ProjectRoot`""
)

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($arguments -join " ")
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$userId = (Get-CimInstance Win32_ComputerSystem).UserName
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Force | Out-Null
Write-Output "Scheduled $TaskName to start in about one minute."
