# Linux SSH 与 Docker 共用 GPU 运行计划

更新时间：2026-09-23

## 适用范围

本文只适用于实验室共用 Linux 服务器 `lab-gpu`：主机名 `seclab03`，地址 `121.48.227.136`，账户 `liangyl`。本文中的 Ubuntu、Linux shell、Docker 组、RTX 4090 和共享 GPU 使用规则都不能套用到新的 Windows 台式机 `desktop-wsl`。

新的 Windows 台式机使用单独的 [台式机 Windows 与 WSL 部署计划](13_台式机Windows与WSL部署计划.md)。在台式机完成只读检查前，不对其 GPU、Docker、WSL 发行版和路径做任何假设。

## 目标

我需要在课题组共用的 Linux GPU 服务器上运行 UniGIR。服务器通过 SSH 登录，所有成员使用 Docker，代码、数据和运行结果通过宿主机目录挂载到容器中。运行流程要满足四个条件：

1. GPU 空闲后才启动任务，不影响其他人的任务；
2. SSH 断开后容器和训练可以继续，日志和 checkpoint 保存在宿主机；
3. 代码、数据、结果和缓存分开保存，实验可以复查；
4. 先完成低成本检查，再决定是否占用 GPU 做 pilot。

项目的研究目标仍是尽快判断 UniGIR 是否值得发展成 CCF-A 主会候选。服务器资源只用于验证关键假设，不自动解锁长训练。

## 当前已知情况

| 项目 | 当前信息 | 影响 |
|---|---|---|
| 本地环境 | Windows，Python 3.13.0，CPU 版 PyTorch，无 CUDA | 不能在本机运行 GPU pilot |
| 原有 GPU 脚本 | `run_scripts/gpu_pilot_s_group.ps1` | 只能作为参数参考，不能直接在 Linux 执行 |
| 原始环境文件 | `environment.yml`，Linux/Conda/Python 3.8.18、PyTorch 2.1.0、CUDA 11.8 | 作为 Docker 基线，仍需与课题组镜像逐项核对 |
| Python 依赖 | `requirements.txt` 中有多项未锁定版本 | 运行前必须记录容器内实际版本 |
| GPU 使用 | 课题组成员共用，需要等待空闲 | 任务要拆成可恢复的小段，不预留长时间独占 |
| Docker | 每个人使用 Docker | 需要核实镜像名、Docker 权限、NVIDIA Container Toolkit 和挂载规则 |
| 官方容器文件 | 公开仓库文件列表中未见 Dockerfile 或 Compose 文件 | Docker 镜像需要使用课题组统一版本，或自行基于 `environment.yml` 构建 |
| 数据协议 | 当前 split 只能作开发数据 | G0 要生成单独的数据根目录并记录数据 seed |

## 原始 InfMasking 仓库核对结果

本地 Git remote 为 `https://github.com/jumaojumao233/InfMasking.git`。公开检索到的论文代码仓库为 [brightest66/InfMasking](https://github.com/brightest66/InfMasking)。官方 README 要求使用 Python 3.8 的 Conda 环境，并通过 `environment.yml` 安装依赖；官方入口包括 `run_scripts/trifeatures_run.sh`、`run_scripts/test.sh`、`run_scripts/multibench_run.sh` 和 `run_scripts/all-mod_run.sh`。

官方环境文件固定了 Python 3.8.18、PyTorch 2.1.0、`pytorch-cuda=11.8`、torchvision 0.16.0、PyTorch Lightning 2.1.1。当前本机的 Python 3.13、CPU 版 PyTorch 和 uv 环境用于 Windows 开发，不能作为服务器 GPU 的环境基线。

公开仓库没有提供 Dockerfile 或 Compose 文件。服务器应优先采用课题组已经验证的、与这套版本接近的统一 Docker 镜像。若课题组没有现成镜像，再基于 `environment.yml` 构建个人镜像，并记录基础镜像 digest、Conda 环境导出、实际 Python/PyTorch/CUDA 版本和安装差异。UniGIR 的 `losses/prototype_alignment.py`、修改后的 `InfMaskingLoss`、配置、测试和 ProjectDocs 必须从当前工作区同步到服务器，不能重新从官方仓库覆盖。

官方 Trifeatures 脚本按 seed=42 到 46 循环运行，官方 MultiBench 脚本也按多个 seed 循环运行。它们适合说明原始实验入口，当前共用 GPU 计划不直接执行整段循环；服务器脚本应拆成一个 seed、一个方法、一个 GPU 的单次运行，并把 `trainer.default_root_dir` 覆盖到挂载的 runs 目录。

## 尚未核实的信息

服务器正式运行前，需要确认：

- SSH 地址、端口、登录账户，以及是否需要跳板机；
- Docker 是否允许普通用户直接执行，是否需要 `sudo`；
- 课题组规定的基础镜像名称、版本和获取方式；
- `nvidia-smi`、Docker GPU 运行时和 `nvidia-container-toolkit` 是否可用；
- 宿主机上可长期保存代码、数据、checkpoint 和日志的目录；
- home 目录和 scratch 目录的容量、清理策略、读写权限；
- 共用 GPU 的使用规则，是否需要在群里登记或通过排队工具申请；
- 是否允许 `--ipc=host`、是否限制 `--shm-size`、是否允许联网下载数据；
- 是否已有统一的数据目录和 Docker 镜像，避免重复占用磁盘。

这些信息没有确认前，我不会假定服务器支持任意 Docker 参数，也不会假定 GPU 空闲状态可以持续。

## 使用 ssh-mcp 建立连接

本次连接采用 [slepp/ssh-mcp](https://github.com/slepp/ssh-mcp)。该工具是一个通过 stdio 提供 MCP 接口的 Python 服务，底层调用本机的 OpenSSH、`scp` 和可选的 `rsync`。它会复用本机 SSH 配置、密钥、SSH Agent 和 ProxyJump，因此服务器别名、账户、端口和跳板机应优先写在本机 SSH 配置中，不把私钥复制到项目目录或 Docker 容器。官方安装方式包括 `uvx --from slepp-ssh-mcp ssh-mcp`，并要求本机有 Python 3.10 以上和 OpenSSH。具体工具、安装方式和已知限制见 [官方 README](https://github.com/slepp/ssh-mcp/blob/main/README.md)。

当前检查发现，本 Codex 会话的工具列表中暂未加载 `ssh-mcp` 服务。因此，安装仓库或运行其命令本身还不能让当前会话自动获得 `ssh_exec` 等 MCP 工具；需要先在实际 MCP 客户端中注册服务，再重新加载会话。服务器目标和认证信息也必须存在，才能进行连接。

本机预检结果：OpenSSH 9.5、`uv` 和 `uvx` 可用；通过临时 Python 3.13 环境启动 `ssh-mcp` 成功，MCP 握手返回服务版本 0.2.0 和 17 个工具。SSH 目录没有 `config` 文件，`known_hosts` 只有 `github.com`，项目文档也没有课题组服务器地址。因此本轮没有发起远端连接。获得服务器目标后，若客户端使用 Codex CLI，可按官方 README 注册：`codex mcp add ssh-mcp -- uvx --from slepp-ssh-mcp ssh-mcp`，然后重新加载 MCP 会话。

### 当前连接目标和安全状态

当前目标为 `liangyl@121.48.227.136:22`，无跳板机信息。第一次连接采用严格主机密钥校验，`ssh-mcp` 已启动成功，但因本机没有该 IP 的已确认 ED25519 主机密钥而停止，未执行任何远端命令。公开扫描得到的指纹为 `SHA256:WqCfqbGXtModqG/8GKax7+I6z1lS14nqtlGRCFkR9SY`，该指纹在与实验室记录核对前不作为可信依据，也不应通过关闭校验的方式绕过。

核对通过后，第一轮只读命令仍限定为主机身份和 GPU 状态检查；若发现其他用户占用 GPU，只记录状态并停止，不启动 Docker 或训练。

### 认证结果

已通过 `ssh-mcp` 尝试使用本机默认 SSH 私钥连接 `liangyl@121.48.227.136:22`。启用 `BatchMode=yes` 后，服务器返回 `Permission denied (publickey,password)`，说明当前私钥不匹配，或该账户还需要密码认证。没有远端命令执行成功。后续应由用户在本机加载正确的 SSH 私钥到 Agent，或手动完成登录；不把密码发送给 Agent，不重复试错认证。认证成功后才进入主机和 GPU 只读预检。

## SSH 公钥配置方案

### 先理解一次登录发生了什么

一次 SSH 登录有两层检查，作用不同。

第一层是确认我连的是哪台服务器。服务器会出示一个主机公钥，客户端把它显示成指纹。当前服务器的 ED25519 指纹是：

```text
SHA256:WqCfqbGXtModqG/8GKax7+I6z1lS14nqtlGRCFkR9SY
```

它回答的是：这是不是实验室那台服务器。它不代表 `liangyl` 的登录权限，也不是登录密码。第一次连接时要核对这个指纹，防止把密码或私钥发给错误的服务器。

第二层是确认登录者是谁。服务器可以用密码确认，也可以用 SSH 公钥确认。密码方式是用户每次输入密码；公钥方式是：

| 内容 | 保存位置 | 作用 | 能否发给 Agent |
|---|---|---|---|
| 私钥 `id_ed25519_lab` | 本机 `C:\Users\breeze\.ssh` | 证明我拥有这把密钥 | 不能 |
| 公钥 `id_ed25519_lab.pub` | 服务器账户的 `~/.ssh/authorized_keys` | 让服务器知道哪些密钥允许登录 | 可以在手动配置时复制 |
| passphrase | 用户记忆或本机 SSH Agent | 解锁本机私钥 | 不能 |

私钥和公钥是一对。登录时，服务器发起一次随机验证，本机用私钥完成证明，服务器用公钥验证。私钥不会传到服务器，密码也不需要交给 `ssh-mcp`。

MobaXterm 的作用是利用当前已经可用的登录方式，完成一次性配置：我先用 MobaXterm 登录，再把公钥添加到自己的服务器账户。之后 MobaXterm 不再是必须的，Windows OpenSSH 和 `ssh-mcp` 都可以用同一把私钥登录。

SSH `config` 只是一个本地快捷配置文件。它把 `lab-gpu` 映射到服务器 IP、用户名、端口和私钥路径，所以以后可以写 `ssh lab-gpu`，不需要每次重复写长地址。它不保存密码，也不会改变服务器。

SSH Agent 是本机内存中的密钥解锁服务。如果私钥设置了 passphrase，Agent 在本机解锁一次，后面的 SSH 命令可以复用；它不会把私钥上传到服务器。若不使用 passphrase，也可以直接指定私钥，但共享电脑上不建议这样做。

`ssh-mcp` 位于最后一层：它接收 Agent 的只读请求，再调用本机 OpenSSH。它不会自动知道 MobaXterm 的密码，也不会绕过服务器认证。官方说明也明确它复用本地 SSH 配置、密钥和 Agent，见 [ssh-mcp README](https://github.com/slepp/ssh-mcp/blob/main/README.md)。

完整路径可以记成：

```text
MobaXterm 临时登录
        ↓
把公钥加入服务器账户
        ↓
Windows 本地保存私钥 + SSH config
        ↓
ssh lab-gpu 先做只读测试
        ↓
ssh-mcp 复用同一套 SSH 配置
        ↓
只读检查 hostname / id / nvidia-smi
```

当前推荐使用一把专用的 Ed25519 密钥。以下步骤需要用户手动完成，Agent 不代为生成私钥，也不接收私钥或 passphrase。

为了减少手动输入，项目提供了 [setup_ssh_access.ps1](../run_scripts/setup_ssh_access.ps1)。它只在用户主动运行时操作本机：生成密钥、显示公钥、复制公钥到剪贴板，并可选追加 SSH config。它不会把私钥打印出来，不会上传公钥，不会保存密码，也不会连接服务器，除非显式加入 `-TestConnection`。

最小运行方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_scripts\setup_ssh_access.ps1 -WriteConfig
```

运行后：

1. 在提示中设置 passphrase；
2. 看到绿色的 `ssh-ed25519 ...` 一行，这就是可以复制到服务器的公钥；
3. 看到 `私钥已经保存到` 的路径，但不要打开或发送该文件；
4. 用 MobaXterm 登录服务器，把公钥加入 `~/.ssh/authorized_keys`；
5. 公钥添加完成后，再运行脚本的只读测试模式：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_scripts\setup_ssh_access.ps1 -KeyName id_ed25519_lab -TestConnection
```

`-TestConnection` 不会重新生成密钥，只会使用已有密钥和 SSH config 执行 `hostname; id`。如果测试失败，不要重复生成密钥或尝试密码，应检查公钥是否完整加入服务器账户，以及本机 passphrase 是否已由 SSH Agent 解锁。

### 当前 ssh-mcp 认证状态

用户已经验证 `ssh lab-gpu` 可以登录，但该命令会在交互式终端中请求私钥 passphrase。`ssh-mcp` 的 BatchMode 不能弹出这个输入框，所以当前返回 `Permission denied (publickey,password)`，远端没有执行任何命令。用户需要在本机 PowerShell 中运行：

```powershell
Start-Service ssh-agent
ssh-add "$env:USERPROFILE\.ssh\id_ed25519_lab"
```

passphrase 只输入到本机 `ssh-add` 提示中，不写入命令或项目。Agent 解锁后，先运行 `ssh -o BatchMode=yes lab-gpu "hostname; id"`，再让 `ssh-mcp` 执行 GPU 只读预检。

### 2026-09-21 只读预检结果

SSH Agent 配置完成后，`ssh-mcp` 已成功通过 `lab-gpu` 连接。远端返回：主机 `seclab03`，Ubuntu 20.04.5，当前账户 `liangyl`，属于 `docker` 组。GPU 0 为 NVIDIA GeForce RTX 4090，总显存 24564 MiB，已用 5360 MiB，利用率 19%，存在一个 Python 计算进程，占用 5358 MiB。

GPU 当前有人使用，本轮按共用服务器规则停止后续检查，没有读取 Docker、磁盘或挂载状态，没有启动容器、写文件、停止进程或训练。下一次只有在 GPU 空闲且符合课题组规则时，才补充宿主机和 Docker 的只读预检。

### 1. 在 Windows 生成专用密钥

在 PowerShell 执行：

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519_lab" -C "liangyl@lab-gpu"
```

建议设置 passphrase，不要覆盖已有的 `id_rsa`。然后复制公钥：

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_lab.pub" | Set-Clipboard
```

剪贴板中只应是以 `ssh-ed25519` 开头的一行公钥。私钥文件 `id_ed25519_lab` 不得复制到项目、Docker 或聊天窗口。

### 2. 通过 MobaXterm 登录后添加公钥

在已经可以登录服务器的 MobaXterm 终端中执行：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
```

把刚才复制的一整行公钥粘贴进去，保存并退出，然后执行：

```bash
chmod 600 ~/.ssh/authorized_keys
```

如果服务器没有 `nano`，可以使用 `vi ~/.ssh/authorized_keys`。只添加自己的公钥，不删除文件中已有的其他公钥。若课题组禁止个人维护 `authorized_keys`，应请管理员按课题组规则配置。

### 3. 在 Windows 创建 SSH 别名

创建或编辑 `C:\Users\breeze\.ssh\config`，写入：

```text
Host lab-gpu
    HostName 121.48.227.136
    User liangyl
    Port 22
    IdentityFile C:/Users/breeze/.ssh/id_ed25519_lab
    IdentitiesOnly yes
```

如果密钥设置了 passphrase，可选用 Windows `ssh-agent` 保存本次登录期间的解锁状态：

```powershell
Start-Service ssh-agent
ssh-add "$env:USERPROFILE\.ssh\id_ed25519_lab"
```

`ssh-add` 会在本机终端中请求 passphrase。不要把 passphrase 发送给 Agent。若 `Start-Service` 因权限受限失败，不要修改系统服务设置，先让用户或管理员按本机策略处理。

### 4. 只读验证

第一次连接时，确认显示的 ED25519 指纹与实验室记录一致：

`SHA256:WqCfqbGXtModqG/8GKax7+I6z1lS14nqtlGRCFkR9SY`

确认后，在 PowerShell 执行：

```powershell
ssh -o BatchMode=yes lab-gpu "hostname; id"
```

该命令只读取主机名和登录身份。如果返回正常，再让 `ssh-mcp` 使用目标 `lab-gpu` 做 GPU 预检。若返回 `Permission denied`，说明公钥尚未生效或密钥没有被 SSH Agent 解锁，不要反复尝试密码。

第一次连接只使用以下只读检查：

```text
date -Is
hostname
uname -a
id
docker --version
docker info
nvidia-smi
df -h
df -i
findmnt
```

检查结果至少记录主机名、登录用户、Docker 版本、Docker 是否可用、GPU 型号和进程、可用空间、项目根目录所在文件系统以及是否存在课题组规定的挂载目录。第一次连接不执行 `docker run`、`docker pull`、`docker build`、`sudo`、数据下载、代码同步和训练。

`ssh-mcp` 支持执行任意远端命令，工具文档也提醒会保存会话转录，转录中可能出现密码、令牌或其他敏感信息。因此连接时不在命令行中输入密码和私钥内容，不把完整 SSH 配置、私钥路径细节和敏感转录复制到项目文档；预检完成后只保留与环境判断有关的结果。

## 宿主机和容器目录

建议在服务器上为本项目准备一个个人根目录，例如 `<SERVER_ROOT>`。实际路径以课题组规定为准。

| 宿主机目录 | 容器目录 | 模式 | 用途 |
|---|---|---|---|
| `<SERVER_ROOT>/source/InfMasking` | `/workspace/InfMasking` | 首次同步可写，正式运行建议只读 | 代码、配置、测试和运行脚本 |
| `<SERVER_ROOT>/data` | `/workspace/InfMasking/dataset/data` | 读写 | Trifeatures、MOSI 和正式确认数据；保持 `dataset/catalog.json` 的相对路径有效 |
| `<SERVER_ROOT>/runs` | `/workspace/runs` | 读写 | TensorBoard、日志、checkpoint 和 manifest |
| `<SERVER_ROOT>/cache` | `/workspace/cache` | 读写 | Torch、Hugging Face 和其他下载缓存 |

代码目录与结果目录分开后，训练命令统一覆盖 `trainer.default_root_dir=/workspace/runs/<experiment>`。这样即使代码挂载为只读，Lightning 日志和 checkpoint 仍能写入宿主机。

不挂载整个 home 目录，不挂载 SSH 私钥，不挂载 Docker socket，也不把 checkpoint 写进容器内部。

## 六个执行阶段

### 0. 本地版本冻结

在上传服务器前，先完成：

1. 运行 6 个单元测试和关键模块语法检查；
2. 保存 `git status --short`、当前提交或差异补丁；
3. 对关键代码、配置和 `ProjectDocs` 生成 SHA256；
4. 记录数据根目录、数据 seed、模型 seed 和计划中的 Docker 镜像；
5. 将本次版本清单与代码一起上传。

当前工作区有大量未提交改动，不能只在服务器执行 `git clone` 后假定得到当前版本。上传方式应选择一个明确版本：提交到个人分支、导出补丁，或打包完整工作区并附带清单。不要用 `git reset --hard` 清理本地改动。

### 1. SSH 登录后的宿主机预检

先在宿主机执行以下检查，结果写入本次运行目录：

```bash
date -Is
hostname
id
docker --version
docker info
nvidia-smi
df -h <SERVER_ROOT>
df -i <SERVER_ROOT>
```

确认 GPU 时至少记录 GPU 编号、型号、显存总量、显存使用量、利用率和当前进程。看到别人的进程时等待，不结束、不暂停、不重置进程。GPU 使用状态应在启动前再次确认，因为空闲状态可能在几分钟内发生变化。

### 2. Docker 镜像和 CUDA 预检

优先使用课题组已经验证过的基础镜像。若必须自己构建镜像，先记录基础镜像 digest，再安装依赖；不要在每次实验的运行容器里临时 `pip install`。

容器启动后，第一步只检查环境：

```bash
python - <<'PY'
import sys
import torch
print(sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
python -m unittest discover -s tests -v
```

`torch.cuda.is_available()`、PyTorch CUDA 版本和宿主机驱动需要同时记录。只看到宿主机有 GPU，不能说明容器已经正确获得 GPU。

### 3. 容器启动和挂载

实际镜像名、用户和路径确认后，使用类似下面的模板。`GPU_ID` 是宿主机空闲 GPU 编号，容器内部通常只看到编号 0。

```bash
docker run --rm -it \
  --name "unigir_${USER}_pilot" \
  --gpus "device=${GPU_ID}" \
  --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HF_HOME=/workspace/cache/huggingface \
  -v "${SERVER_ROOT}/source/InfMasking:/workspace/InfMasking:ro" \
  -v "${SERVER_ROOT}/data:/workspace/InfMasking/dataset/data:rw" \
  -v "${SERVER_ROOT}/runs:/workspace/runs:rw" \
  -v "${SERVER_ROOT}/cache:/workspace/cache:rw" \
  -w /workspace/InfMasking \
  "${IMAGE}" bash
```

`--ipc=host` 只在课题组允许时加入。容器若以 root 写入挂载目录，会造成后续权限问题；优先使用与宿主机 UID/GID 对应的普通用户运行，或先确认镜像的默认用户。

长任务在宿主机的 `tmux` 或 `screen` 会话中启动。SSH 断开后重新连接，再进入同一会话查看日志。容器退出后，结果仍应保留在 `/workspace/runs`。

### 4. 容器内低成本冒烟

按以下顺序执行：

1. 单元测试；
2. Hydra 配置解析；
3. S 组、`max_size=64`、1 epoch、关闭 probing 的 GPU 冒烟；
4. 检查显存、运行时间、日志、checkpoint 和 TensorBoard；
5. 确认 Baseline 与 UniGIR 的配置只在 profile 分支上不同。

所有输出写到独立的 `/workspace/runs/<experiment>`，不使用项目目录下的默认输出目录。

### 5. G0 数据和版本固定

正式 pilot 前，使用新的数据根目录生成未参与旧开发选择的数据。数据 seed 和模型 seed 分开记录，例如：

```text
/workspace/InfMasking/dataset/data/trifeatures_confirm_data_seed_20260921/
data_seed=20260921
model_seed=42
```

需要保存：

- train/test 文件数量和目录结构；
- 每个数据文件 SHA256；
- 当前代码提交或差异补丁；
- 关键文件 SHA256；
- Docker 镜像名称和 digest；
- Python、PyTorch、CUDA、驱动和 GPU 型号；
- 完整 Hydra 配置；
- 运行命令、开始时间、结束时间和退出码。

G1 pilot 仍可以使用当前开发 split 做低成本信号检查，但它不能被写成最终测试结果。正式确认阶段必须使用独立数据根目录。

### 6. G1 单 seed pilot

现有 PowerShell 脚本不能直接使用。服务器上应使用 Linux shell 脚本，并把一次运行限制为一个 seed、一个方法和一个 GPU。建议顺序：

| 顺序 | 实验 | 目的 |
|---:|---|---|
| 1 | seed=42，InfMasking Baseline | 检查服务器速度、显存和输出路径 |
| 2 | seed=42，UniGIR 单向，$K=128$、queue=1024、$\alpha=0.25$ | 检查修复后是否仍有信号 |
| 3 | 检查两组结果和资源成本 | 决定是否继续 seed=7、123 |
| 4 | seed=7、123 的配对运行 | 只在前两组正常且信号没有明显退化时进行 |

每个 run 固定 10 epoch，训练结束后只做一次最终 probing。不要在同一时间启动 Baseline 和 UniGIR，不使用 DDP，不抢占其他人的 GPU。

## GPU 空闲窗口的使用规则

- 启动前查看一次 `nvidia-smi`，真正启动容器前再查看一次；
- 只选择没有其他进程、且符合课题组规则的 GPU；
- 训练期间持续记录显存和利用率；
- 发现他人任务出现、显存异常增长或服务器管理员要求释放时，按约定停止；
- 一次只运行一个实验，先完成 seed=42 的 Baseline 和 UniGIR，再申请下一段窗口；
- 每次任务结束后立即保存 manifest、日志、checkpoint 路径和退出码。

## 暂不执行的事项

- 不直接运行现有 PowerShell GPU 脚本；
- 不在容器里临时安装一套未记录的依赖；
- 不把整个项目目录、home 目录或 Docker socket 作为无边界挂载；
- 不在没有 GPU 空闲确认时启动训练；
- 不从 G1 直接跳到 100 epoch；
- 不把服务器上的第一组结果直接写成论文正式结果。

## 下一步

先获得服务器的 SSH、Docker、镜像、GPU 和存储信息，再补两个 Linux 侧工具：GPU 预检脚本和单 run 可恢复的 pilot 脚本。工具通过容器内冒烟后，才进入 G0 数据固定和 G1 seed=42。
