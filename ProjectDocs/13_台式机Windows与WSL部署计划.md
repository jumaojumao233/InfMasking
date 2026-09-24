# 台式机 Windows 与 WSL 部署计划

更新时间：2026-09-24

## 当前实际状态

用户已配置 `winpc` 别名，连接成功。主机为 `DESKTOP-STVGT1D`，账户为 `desktop-stvgt1d\liangyelian`，系统为 Windows 10 `10.0.19045.3693`。GPU 为 NVIDIA GeForce GTX 1080 Ti，显存 11264 MiB，驱动 536.23。WSL 功能和 VirtualMachinePlatform 已启用，但没有发行版；通过 GitHub 下载 Ubuntu 时发生 `WININET_E_TIMEOUT`，当前实验使用 Windows 原生 Python，不再把 WSL 作为本轮阻塞点。

代码已部署到 `C:\Users\liangyelian\work\InfMasking_desktop_20260923`，S 组 `trifeatures_3combi` 数据已单独同步并核对为 2600 个文件。远程虚拟环境使用 Python 3.11.15、PyTorch `2.1.0+cu118`、PyTorch Lightning 2.1.1、TorchMetrics 1.2.0、TensorBoard 2.14.1、timm 0.9.12 和 NumPy 1.26.4；`torch.cuda.is_available()` 为 `True`。

修复版 seed=42、7、123 的 Baseline 和 UniGIR 已完成 10 epoch S 组配对 pilot，日志已复制到本地 `outputs/winpc_20260924_s42/`、`outputs/winpc_20260924_s7_baseline/`、`outputs/winpc_20260924_s7_unigir/`、`outputs/winpc_20260924_s123_baseline/` 和 `outputs/winpc_20260924_s123_unigir/`。三 seed 中有 2 个 seed 的 synergy ROC AUC 提升，平均变化约 +0.002，四任务平均 acc@1 平均变化约 +0.014，达到 Weak Go；这些结果仍属于 development pilot，不能作为最终论文结果。

断点续跑已经通过真实 smoke 验证。seed=7 Baseline 和 UniGIR 均已完成 10 epoch 和四项 probing。Baseline 四任务平均 acc@1/ROC AUC=0.493/0.817，UniGIR=0.482/0.816；两个方法均已保存独立 `last.ckpt`。当前 `nvidia-smi` 只显示一张 GTX 1080 Ti，因此实验按同一 GPU 串行运行，不安排第二个并行任务。

Windows 长任务不能依赖 SSH 前台会话。一次性任务计划的定时触发方式已验证可用；后续 G2 机制对照继续一次只启动一个方法和一个任务，完成后保留日志与 checkpoint，并清理本任务创建且已完成的计划任务。

## 与 `lab-gpu` 的边界

当前目标是用户的新 Windows 台式机 `desktop-wsl`，不是上次的实验室 Linux 服务器 `lab-gpu`，也不能按 RTX 4090 机器处理。`lab-gpu` 的主机名是 `seclab03`，地址是 `121.48.227.136`；本台式机地址是 `172.16.1.113`。两台机器的账户、操作系统、GPU、Docker、文件路径和资源使用规则分别核实。

目前只知道台式机运行 Windows 且已安装 WSL。台式机是否有 NVIDIA GPU、GPU 型号和显存、是否启用 WSL2、使用哪个发行版、是否安装 Docker、是否支持 GPU 透传，都未核实。后续必须以台式机只读检查结果为准。

## 初次连接阶段的历史记录

下面保留台式机首次配置 SSH 时的中间状态。当前连接和实验状态以上面的“当前实际状态”为准。

本机到 `172.16.1.113` 的 Ping 成功，TCP 22 可达；非交互 SSH 返回 `Permission denied (publickey,password,keyboard-interactive)`。当前还没有取得台式机账户的公钥认证，因此没有执行任何远端读取、代码同步、环境安装或实验命令。

前一版建议错误地复用了实验室 `lab-gpu` 专用密钥 `id_ed25519_lab`。台式机改用独立的 `id_ed25519_desktop_wsl` 密钥：只把对应的 `.pub` 公钥登记到台式机账户，现有实验室私钥和公钥都不复制、不修改、不复用。认证成功后才读取 Windows、WSL、Python、Docker、GPU 和磁盘状态。

## 当前认证排查

本机独立公钥指纹为 `SHA256:Ab4NcETaORy3jj9JEQdcQOffjKXByjpYJu6addIBi6E`。调试日志确认 SSH 客户端已经把这把公钥发送到台式机，但远端拒绝认证。需要在台式机已有的密码 SSH 会话中确认当前账户和实际用户目录，不要手动猜写路径：

```powershell
whoami
$env:USERNAME
$env:USERPROFILE
Get-ChildItem -Force (Join-Path $env:USERPROFILE ".ssh")
ssh-keygen -lf (Join-Path $env:USERPROFILE ".ssh\authorized_keys")
```

如果 `authorized_keys` 不存在，说明文件名或目录仍不正确。如果指纹与上面的本机指纹不一致，说明粘贴的是另一把公钥。若当前账户属于 Windows Administrators 组，再检查 `C:\ProgramData\ssh\administrators_authorized_keys`；未确认前不修改该系统级文件。

## 目标

把当前 InfMasking/UniGIR 工作区迁移到用户自己的 Windows 台式机和 WSL 环境，用于代码预检、依赖验证和小规模实验。实验室共用 Linux GPU 服务器仍然用于需要 GPU 的正式 pilot，不能把两套环境混成一个结果来源。

当前目标信息：

| 项目 | 当前信息 | 影响 |
|---|---|---|
| 目标主机 | `172.16.1.113` | 通过蒲公英网络访问，地址是否长期稳定需要后续记录 |
| 登录账户 | `liangyelian` | 只使用该账户，不使用管理员账户执行项目命令 |
| 操作系统 | Windows，已安装 WSL | 需要区分 Windows shell 和 WSL shell；WSL 版本和发行版待核实 |
| 与 `lab-gpu` 的关系 | 新的独立目标 | 不继承 `lab-gpu` 的 RTX 4090、Linux、Docker 或共享 GPU 结论 |
| 当前认证 | 用户已确认密码可以手动登录 | 不把密码写入命令、脚本或 MCP 转录 |
| 推荐认证 | SSH 公钥和本机 SSH Agent | 供 `ssh-mcp` 进行非交互只读操作 |
| 目标代码 | 当前工作区的 UniGIR 修改版 | 同步前生成版本清单，不从官方仓库覆盖本地修改 |

## 先配置台式机公钥

不要把密码写入脚本或发给 Agent。先在当前这台 Codex 电脑的 PowerShell 中复制公钥：

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519_desktop_wsl" -C "liangyelian@desktop-wsl"
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_desktop_wsl.pub" | Set-Clipboard
```

然后用你已经可以登录的密码 SSH 会话进入台式机，把剪贴板中的整行 `ssh-ed25519 ...` 公钥加入对应账户。

如果 SSH 进入的是 Windows PowerShell，执行：

```powershell
$sshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
notepad (Join-Path $sshDir "authorized_keys")
```

把公钥粘贴成一整行并保存，然后关闭记事本。这个文件属于 `liangyelian` 自己的账户，不要删除其中已有的其他公钥。

如果 SSH 进入的是 WSL Linux shell，执行：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

同样只添加公钥整行。然后在当前本机 `C:\Users\breeze\.ssh\config` 追加一个别名：

```text
Host desktop-wsl
    HostName 172.16.1.113
    User liangyelian
    Port 22
    IdentityFile C:/Users/breeze/.ssh/id_ed25519_desktop_wsl
    IdentitiesOnly yes
```

再执行只读测试：

```powershell
ssh -o BatchMode=yes desktop-wsl "hostname; whoami"
```

测试成功后，Agent 才开始执行后续只读环境检查。若 Windows OpenSSH 使用的是管理员专用公钥文件，普通用户目录配置不生效，此时先不要修改系统文件，把测试报错发给我，由我根据实际 sshd 配置判断路径。

## 为什么先配置公钥

当前密码登录适合用户在终端中手动输入，不适合让 Agent 自动操作。`ssh-mcp` 复用本机 OpenSSH，非交互命令不能安全接收密码，且会话转录可能保存敏感输入。因此先把本机新生成的 `id_ed25519_desktop_wsl.pub` 加入台式机账户，私钥仍留在本机，后续通过 SSH Agent 认证。实验室的 `id_ed25519_lab` 只用于 `lab-gpu`。

用户已经可以用密码登录台式机时，在台式机的 SSH 会话中完成公钥登记。公钥登记前不进行代码同步；不把私钥复制到台式机，不把密码写进项目。

## 部署阶段

### 1. SSH 公钥认证

先让 `ssh -o BatchMode=yes desktop-wsl "hostname; whoami"` 成功。第一次连接仍需核对主机指纹。目标别名建议使用 `desktop-wsl`，避免和实验室 Linux GPU 服务器的 `lab-gpu` 混淆。

现场操作顺序固定为：

1. 在当前电脑的 PowerShell 生成并复制 `id_ed25519_desktop_wsl.pub`；
2. 在台式机已有密码 SSH 会话中添加公钥；
3. 在当前电脑的 `C:\Users\breeze\.ssh\config` 中加入 `desktop-wsl`；
4. 从当前电脑运行 `ssh -o BatchMode=yes desktop-wsl "hostname; whoami"`；
5. 认证成功后停止手动密码登录，后续由 SSH Agent 和 `ssh-mcp` 复用公钥。

这一步不需要把密码发给 Agent。若 `ssh -o BatchMode=yes` 失败，只检查公钥位置、账户和 Agent，不重复尝试密码。

### 2. 只读环境检查

连接成功后，先判断 SSH 默认 shell 和 WSL 入口：

```text
hostname
whoami
ver
wsl.exe --status
wsl.exe --list --verbose
where.exe python
where.exe docker
where.exe nvidia-smi
```

如果 SSH 默认进入 PowerShell，WSL 命令必须显式通过 `wsl.exe` 调用。之后再在目标发行版内读取：

```text
cat /etc/os-release
python3 --version
python3 -m pip --version
docker --version
nvidia-smi
df -h
```

这些命令只读取状态。当前不能假定 Docker Desktop 已安装、WSL2 已启用、发行版名称固定或台式机有 NVIDIA GPU。

### 3. 代码打包和同步

代码同步前保留当前工作区版本清单，至少记录 Git 状态、关键文件 SHA256、配置和运行入口。首批同步内容包括：

- `main_*.py`、`models/`、`losses/`、`dataset/`、`evaluation/`、`pl_modules/`、`utils.py`；
- `configs/`、`tests/`、`run_scripts/`；
- `environment.yml`、`requirements.txt`、`README.md`；
- `ProjectDocs/` 和当前研究记录。

默认排除：`.venv/`、`.git/`、`__pycache__/`、`outputs/`、`diagnostic_logs/`、`overnight_logs/`、`probe_logs/`、`seed7_logs/`、`seed7_noqueue_logs/`、`gpu_pilot_logs/`、大型 checkpoint 和本地临时缓存。数据目录单独确认大小、路径和是否需要同步，不随第一批代码包盲目复制。

目标路径不能直接假定。优先在 WSL 的 Linux 文件系统中建立用户目录，例如 `~/work/InfMasking`；如果 Docker Desktop 需要从 Windows 路径挂载，再根据实际 WSL 配置选择 Windows 目录。正式运行前记录宿主机路径、WSL 路径和容器路径的对应关系。

### 4. 环境建立

优先顺序：

1. 先复用用户已有的 WSL Python/Conda/Docker 环境；
2. 如果没有环境，再基于 `environment.yml` 或课题组镜像建立环境；
3. 不在每次运行时临时安装依赖；
4. 记录 Python、PyTorch、CUDA、驱动、Docker 和 WSL 版本。

Windows 本地 `.venv` 不能直接复制到 WSL。WSL 需要重新建立 Linux 环境，不能把 Windows 路径、Windows Python 可执行文件或本机 CPU 版 PyTorch 当成 WSL GPU 环境。

### 5. 低成本验证

环境建立后按顺序执行：

1. Python 导入和六个单元测试；
2. Hydra 配置解析；
3. CPU 或 GPU 设备可用性检查；
4. S 组 `max_size=64`、1 epoch、关闭 probing 的冒烟；
5. 检查日志和 checkpoint 是否写入用户指定目录。

冒烟通过后，才讨论 10 epoch pilot。正式结果仍遵循 `ProjectDocs/08_AGENT_CONTINUATION.md` 中的 G0—G4 规则。

## 安全边界

- 不在项目文件中保存台式机密码；
- 不把私钥复制到台式机、WSL、Docker 或代码包；
- 不使用 `sudo`、管理员账户、Docker socket 或无边界挂载；
- 不覆盖目标目录中已有文件，第一次同步使用新目录或显式版本目录；
- 不删除、不停止、不暂停台式机上已有进程；
- 不在环境检查前启动训练；
- 发现台式机已有 GPU 任务时，只记录状态，等待用户确认。

## 当前阻塞

台式机目前只有密码登录信息，`ssh-mcp` 不能安全地把密码作为非交互输入。前一版计划错误地引用了 `lab-gpu` 的 `id_ed25519_lab`，已经改正。下一步需要用户在本机生成 `id_ed25519_desktop_wsl`，通过已经可用的密码 SSH 会话把对应的 `.pub` 公钥加入台式机账户。公钥认证成功后，再由 Agent 连接并继续只读检查。
