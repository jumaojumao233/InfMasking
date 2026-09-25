param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputRoot = Join-Path $root "outputs\scheduled_g5_0430"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$codexRoot = Join-Path $env:LOCALAPPDATA "OpenAI\Codex\bin"
$codex = Get-ChildItem -LiteralPath $codexRoot -Recurse -Filter "codex.exe" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $codex) {
    throw "Codex CLI was not found under $codexRoot"
}

$promptPath = Join-Path $PSScriptRoot "g5_0430_analysis_prompt.txt"
$runLog = Join-Path $outputRoot "codex_0430.run.log"
$lastMessage = Join-Path $outputRoot "codex_0430.last_message.md"

$env:CODEX_HOME = Join-Path $env:USERPROFILE ".codex"
Get-Content -LiteralPath $promptPath -Raw |
    & $codex.FullName exec `
        -C $root `
        --ask-for-approval never `
        -s workspace-write `
        -c 'model="gpt-5.6-luna"' `
        -c 'model_reasoning_effort="max"' `
        -o $lastMessage `
        -
    2>&1 | Tee-Object -LiteralPath $runLog

exit $LASTEXITCODE
