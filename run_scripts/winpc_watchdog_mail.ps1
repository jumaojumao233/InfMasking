<#
    Shared mail body formatter for winpc watchdogs.

    Completed run names are intentionally omitted from periodic mail. The
    status JSON still contains the full queue for later inspection.
#>

function Get-WatchdogNotificationBodyLines {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Result
    )

    $runs = @($Result.Runs)
    $completed = @($runs | Where-Object { [string]$_.State -eq "Succeeded" })
    $active = @($runs | Where-Object { [string]$_.State -in @("Running", "Pending") })
    $blocked = @($runs | Where-Object {
        [string]$_.State -in @("Failed", "CheckpointMissing")
    })
    $next = @($runs | Where-Object { [string]$_.State -eq "Missing" } |
        Sort-Object { [int]$_.Order } | Select-Object -First 1)
    $stopReasons = @($Result.StopReasons | Where-Object {
        -not [string]::IsNullOrWhiteSpace([string]$_)
    })

    $bodyLines = @()
    $bodyLines += "Timestamp: $($Result.Timestamp)"
    $bodyLines += "Host: $($Result.Host)"
    if (-not [string]::IsNullOrWhiteSpace([string]$Result.Experiment)) {
        $bodyLines += "Experiment: $($Result.Experiment)"
    }
    $bodyLines += "Action: $($Result.Action)"
    $bodyLines += "Detail: $($Result.Detail)"
    $bodyLines += ""
    $bodyLines += "Run summary:"
    $bodyLines += "  Configured: $($runs.Count)"
    $bodyLines += "  Completed: $($completed.Count)"
    $bodyLines += "  Active or pending: $($active.Count)"
    $bodyLines += "  Failed or blocked: $($blocked.Count)"

    if ($active.Count -gt 0) {
        $bodyLines += ""
        $bodyLines += "Current run(s):"
        foreach ($run in $active) {
            $bodyLines += ("  {0}. {1} seed={2} state={3} detail={4}" -f `
                $run.Order, $run.Method, $run.Seed, $run.State, $run.Detail)
        }
    }
    elseif ($next.Count -gt 0) {
        $bodyLines += ""
        $bodyLines += "Next run:"
        $bodyLines += ("  {0}. {1} seed={2} state={3} detail={4}" -f `
            $next[0].Order, $next[0].Method, $next[0].Seed, $next[0].State, $next[0].Detail)
    }
    elseif ($Result.Action -eq "ALL_COMPLETE") {
        $bodyLines += ""
        $bodyLines += "Completion: all configured runs succeeded and have checkpoints."
    }

    if ($blocked.Count -gt 0) {
        $bodyLines += ""
        $bodyLines += "Failed or blocked run(s):"
        foreach ($run in $blocked) {
            $bodyLines += ("  {0}. {1} seed={2} state={3} detail={4}" -f `
                $run.Order, $run.Method, $run.Seed, $run.State, $run.Detail)
        }
    }

    if ($stopReasons.Count -gt 0) {
        $bodyLines += ""
        $bodyLines += "Stop reasons:"
        $bodyLines += $stopReasons
    }

    if ($null -ne $Result.Gpu) {
        $bodyLines += ""
        $bodyLines += "GPU check: safe=$($Result.Gpu.Safe); detail=$($Result.Gpu.Detail)"
    }

    if ($completed.Count -gt 0) {
        $bodyLines += ""
        $bodyLines += "Completed run names are omitted from this mail; inspect the watchdog status JSON for the full queue history."
    }

    return $bodyLines
}
