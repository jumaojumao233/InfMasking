[CmdletBinding()]
param(
    [string]$KeyName = "id_ed25519_lab",
    [string]$HostAlias = "lab-gpu",
    [string]$HostName = "121.48.227.136",
    [string]$UserName = "liangyl",
    [int]$Port = 22,
    [switch]$WriteConfig,
    [switch]$TestConnection
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($KeyName -notmatch "^[A-Za-z0-9._-]+$") {
    throw "KeyName 只能包含字母、数字、点、下划线和短横线。"
}

if ($HostAlias -notmatch "^[A-Za-z0-9._-]+$") {
    throw "HostAlias 只能包含字母、数字、点、下划线和短横线。"
}

if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port 必须在 1 到 65535 之间。"
}

$sshKeygen = Get-Command ssh-keygen -ErrorAction Stop
$ssh = Get-Command ssh -ErrorAction Stop
$sshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null

$privateKeyPath = Join-Path $sshDir $KeyName
$publicKeyPath = "$privateKeyPath.pub"
$configPath = Join-Path $sshDir "config"

if ($TestConnection) {
    if (-not (Test-Path -LiteralPath $privateKeyPath) -or -not (Test-Path -LiteralPath $publicKeyPath)) {
        throw "TestConnection 找不到完整密钥，请先不带 -TestConnection 运行脚本生成密钥。"
    }
}
else {
    if ((Test-Path -LiteralPath $privateKeyPath) -or (Test-Path -LiteralPath $publicKeyPath)) {
        throw "拒绝覆盖已有密钥：$privateKeyPath 或 $publicKeyPath。请换一个 KeyName，或先由你确认后手动处理旧文件。"
    }

    Write-Host "即将生成一对 Ed25519 密钥。"
    Write-Host "私钥位置：$privateKeyPath"
    Write-Host "公钥位置：$publicKeyPath"
    Write-Host "接下来 ssh-keygen 会请求设置 passphrase。建议设置，不要把 passphrase 发给任何人。"

    & $sshKeygen.Source -t ed25519 -f $privateKeyPath -C "$UserName@$HostAlias"
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-keygen 生成密钥失败，退出码：$LASTEXITCODE。"
    }

    if (-not (Test-Path -LiteralPath $privateKeyPath) -or -not (Test-Path -LiteralPath $publicKeyPath)) {
        throw "ssh-keygen 返回成功，但没有找到预期的密钥文件。"
    }
}

$publicKey = (Get-Content -LiteralPath $publicKeyPath -Raw).Trim()
if ($publicKey -notmatch "^ssh-ed25519\s+\S+(\s+.*)?$") {
    throw "生成的公钥格式异常，请不要复制它，并联系项目负责人检查。"
}

try {
    Set-Clipboard -Value $publicKey
    $clipboardStatus = "已复制到剪贴板"
}
catch {
    $clipboardStatus = "未能自动复制到剪贴板，请从公钥文件中复制"
}

$fingerprint = (& $sshKeygen.Source -lf $publicKeyPath -E sha256 2>&1 | Out-String).Trim()

Write-Host ""
Write-Host "生成完成。"
Write-Host ""
Write-Host "公钥可以复制到服务器，内容如下：" -ForegroundColor Green
Write-Host $publicKey -ForegroundColor Green
Write-Host "公钥状态：$clipboardStatus"
Write-Host "公钥指纹：$fingerprint"
Write-Host ""
Write-Host "私钥已经保存到：$privateKeyPath" -ForegroundColor Yellow
Write-Host "不要打开、复制、上传或发送私钥文件。"

if ($WriteConfig) {
    $hostBlock = @"

Host $HostAlias
    HostName $HostName
    User $UserName
    Port $Port
    IdentityFile $($privateKeyPath.Replace("\", "/"))
    IdentitiesOnly yes
"@

    $existingConfig = if (Test-Path -LiteralPath $configPath) {
        Get-Content -LiteralPath $configPath -Raw
    }
    else {
        ""
    }

    $hostPattern = "(?m)^\s*Host\s+$([regex]::Escape($HostAlias))\s*$"
    if ($existingConfig -match $hostPattern) {
        throw "SSH config 中已经存在 Host $HostAlias。脚本不会覆盖已有配置。"
    }

    Add-Content -LiteralPath $configPath -Value $hostBlock -Encoding utf8
    Write-Host "已把 Host $HostAlias 添加到：$configPath"
}
else {
    Write-Host ""
    Write-Host "尚未修改 SSH config。需要时重新运行并加入 -WriteConfig。"
}

Write-Host ""
Write-Host "下一步：用 MobaXterm 登录服务器，把上面的公钥整行加入 ~/.ssh/authorized_keys。"
Write-Host ('完成后再运行：ssh -o BatchMode=yes {0} "hostname; id"' -f $HostAlias)

if ($TestConnection) {
    if (-not $WriteConfig -and -not (Test-Path -LiteralPath $configPath)) {
        throw "TestConnection 需要先生成 SSH config。请同时使用 -WriteConfig。"
    }

    Write-Host "开始只读测试 SSH 连接：hostname; id"
    & $ssh.Source -o BatchMode=yes $HostAlias "hostname; id"
    if ($LASTEXITCODE -ne 0) {
        throw "SSH 只读测试失败，退出码：$LASTEXITCODE。没有执行其他远端操作。"
    }
}
