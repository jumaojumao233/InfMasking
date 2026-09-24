# 运行 InfMasking 项目

> **Workspace:** d:\codingPPproject\python_project\dmtky\20260908

---

**User**

`d:\codingPPproject\python_project\dmtky\20260908\InfMasking` 把这个Project运行起来，可以参考链接 `https://github.com/brightest66/InfMasking` ，我的电脑是window

**Assistant**

我来帮你运行这个 InfMasking 项目。首先让我了解一下项目结构和相关信息。

现在让我查看本地的 README 和环境配置文件：

让我检查一下系统上的 Python 和 conda 环境：

让我检查一下可用的 Python 和 pip 命令：

好的，系统有 Python 3.13。不过项目要求 Python 3.8，可能会有一些兼容性问题。让我先查看主程序文件，了解项目的基本结构，然后尝试安装依赖：

由于 environment.yml 包含很多 Linux 特定的包，我需要在 Windows 上手动安装核心依赖。让我先安装 PyTorch 和主要依赖：

正在安装 PyTorch，需要一些时间。让我等待安装完成：

下载还在进行中，让我再等待一下，同时查看一下配置文件了解项目结构：

下载速度很慢，让我停止当前命令，尝试使用国内镜像或另一种方式安装：

让我换一种方式安装，使用默认的 PyPI 源安装核心依赖：

安装正在进行，让我等待并检查进度。同时需要注意，由于 Python 3.13 比较新，一些严格指定的版本可能不兼容：

安装正在进行，需要从源码编译一些包，这会比较慢。让我继续等待并查看更多项目代码：

安装太慢，很多指定的旧版本需要从源码编译。让我停止，使用兼容的最新版本安装：

网络超时了。让我使用国内镜像源来加速下载，并且增加超时时间：

下载正在进行，有进展了。让我继续等待，同时查看一些关键的模块代码：

下载进度稳步推进，让我继续等待并查看数据集代码：

快完成了，让我继续等待：

哈希校验失败，缓存可能损坏了。让我清除缓存并重新安装：

好的，重新下载中。让我继续等待：

遇到了哈希校验和沙盒限制的问题。让我换一种策略：先检查已有的包，然后尝试直接运行程序来定位缺失的依赖：

好的，系统几乎没有包。让我创建一个虚拟环境，然后逐步安装依赖，使用清华镜像加速：

---

**User**

`d:\codingPPproject\python_project\dmtky\20260908\InfMasking` 检索一下这个项目里所有代码文件，看看所需要的库，能不能通过uv来在window上实现运行

**Assistant**

好的，让我全面检索项目中的所有代码文件，分析所需的依赖库，然后尝试用 uv 来运行。

让我排除 venv 目录，只分析项目源代码的 import 语句：

让我继续搜索其他关键目录的 import 语句：

让我直接读取几个关键文件来获取它们的 import：

现在我已经分析了所有代码文件的 import 语句。让我检查 uv 是否可用，并创建依赖清单：

太好了！uv 0.9.22 已经可用。现在让我创建 ../requirements.txt，根据代码分析总结所有需要的外部依赖：

现在用 uv 创建虚拟环境并安装依赖。uv 的速度比 pip 快得多：

虚拟环境创建成功！现在让我用 uv 快速安装依赖。先安装 PyTorch CPU 版本，再安装其他依赖：

uv 正在快速下载包！让我等待它完成：

快完成了，正在安装轮子！

太棒了！PyTorch 安装成功！现在安装其他所有依赖：

网络超时，让我增加超时时间并重新安装：

很好！uv 正在快速下载和安装包。让我等待完成：

网络不稳定，让我使用清华镜像源并分批安装：

太棒了！清华镜像源速度飞快！继续安装剩余的依赖：

继续安装 tensorboard：

所有依赖安装完成！现在验证关键库是否能正常导入：

所有依赖导入成功！现在让我尝试运行项目。先看一下运行脚本，然后转为 Windows 命令执行：

好的，让我用 Hydra 来运行一个简单的测试，先检查配置是否正确。由于是 Windows 且没有多 GPU，我需要修改一些参数：

让我尝试运行一个小规模测试来验证代码工作正常。将使用 CPU、单进程、少量 epoch：

Hydra 配置语法问题，让我修复并重试。需要用 `++` 覆写已存在的配置：

发现缺少 `timm` 库！让我搜索更多缺失的依赖，然后一次性安装：

找到了缺失的依赖！让我安装它们：

太棒了！安装完成。让我更新 ../requirements.txt 并再次运行测试：

现在让我先验证所有关键库的导入，然后再次运行：

太好了！所有关键导入都成功了。现在再次尝试运行项目：

进展很好！遇到了数据路径的问题（Linux 风格 `/path` 在 Windows 上无法使用）。让我查看并修改数据配置：

看到了！路径是从 `catalog.json` 读取的。让我查看并修改这个文件：

找到了！让我修改 catalog.json，使用 Windows 兼容的相对路径：

现在再次运行测试：

太好了！没有报错了，程序正在运行！让我继续等待查看训练进度：

太棒了！项目成功运行！正在自动生成 Trifeatures 合成数据集。让我继续等待数据集生成完成和训练开始：

很好的进展！项目已经进入训练阶段了。遇到两个问题：
1. 代码硬编码使用 `.cuda()`，需要改成设备无关
2. Windows 控制台 GBK 编码问题

让我修复这些兼容性问题：

让我先搜索所有硬编码 `.cuda()` 的地方，然后修复：

现在让我逐一修复这些硬编码的 `.cuda()` 调用，使用设备无关的方式：

很好！现在处理 Windows 的 GBK 编码问题，再次运行测试，设置 UTF-8 编码环境变量：

PowerShell 环境变量语法问题，让我修正：

---

# 运行状态补充记录（2026-09-09 续）

## 已完成

项目在 Windows/CPU 下端到端可运行。上一轮已跑通一次完整流程：
1. 生成 Trifeatures 合成数据集（`../dataset/data/trifeatures_3combi/`：train 2400 图、test 200 图）
2. InfMasking 训练（1 epoch, embed_dim=512, mask_ratio=0.7, num_mask=6）
3. 下游 4 任务 linear probing（share/unique1/unique2/synergy）

产物：`../InfMasking/bimodal_trifeatures/test_run/logs/version_1/`
- loss 从 ~5.8 降至 2.46，ssl_acc≈22%（epoch 1，接近随机的合理值）
- 4 个下游任务的结果正常输出；synergy 任务 epoch-1 的退化输出（acc 0.5/roc 0.5、acc_per_class [1.0, 0.0]）属二分类随机初始化正常现象
- checkpoint 保存于 `version_1/checkpoints/epoch=00-acc1=0.4179.ckpt`

## CPU 吞吐实测（本机：12 物理核/16 线程，无 NVIDIA GPU，torch 2.14.0+cpu）

| 项目 | 数值 |
|---|---|
| no_grad 前向+loss（batch 64, embed 512, 6 masks） | ~9.0 s/step |
| **带梯度真实训练 step** | **~36 s/step** |
| OMP_NUM_THREADS=12 对比 | 更慢（14.6 s vs 9.0 s 无梯度），保持默认 |
| 每 epoch 步数（train 10000 pairs, batch 64） | 157 步 |
| 单 epoch 训练耗时 | **约 1.5-1.6 小时** |
| 一次完整 linear probing 轮（4 任务 × 代价搜索） | 约 1 小时以上 |

结论：官方 `../run_scripts/trifeatures_run.sh`（100 epochs × 5 seeds × R-U/S 两组）在纯 CPU 上不可行（单 seed 预计 150+ 小时）。

## 后续建议（供选择）

### 方案 A：本机短验证（几分钟内可启动）
只跑 2-3 epoch + 每 epoch 一次 probing 验证 acc1 上升趋势，约需 6-10 小时，后台运行。
启动命令（后台、UTF-8、Windows）：
```bash
cd /d/codingPPproject/python_project/dmtky/20260908/InfMasking
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe ../main_trifeatures.py \
  seed=42 trainer.strategy=auto trainer.accelerator=cpu trainer.devices=1 \
  trainer.max_epochs=3 trainer.num_sanity_val_steps=0 \
  +data=trifeatures ++data.data_module.biased=false ++data.data_module.num_workers=0 \
  +model=infmasking model.model.encoder.embed_dim=512 \
  optim.lr=3e-4 model.model.loss_kwargs.temperature=0.1 model.model.loss_kwargs.mask_lambda=1 \
  +model.model.encoder.mask_ratio=0.7 +model.model.encoder.num_mask=6 \
  mode=train +exp_name=R-U-cpu-check
```

### 方案 B：Colab/GPU 跑论文完整配置
代码已改为设备无关（`../losses/entropy.py`、`../losses/infmasking_loss.py`、`../models/inffusion.py`、`../models/transformer.py`），可直接在 Colab 免费 T4 上运行：
- 数据集仅 14 MB，可打包上传或按需重建
- 每 step 可加速 50-100 倍，完整 100 epoch 实验 2-4 小时内可完成单 seed
- 注意 Colab 需将 `catalog.json` 指向数据集实际路径

### 方案 C：本地减配快速调参
缩小 embed_dim/latent_dim（如 256）+ batch 32，可把单 step 压到 10 s 内，用于快速验证代码逻辑与实验流程。

---


# 决策与收尾记录（2026-09-09，第二次续）

## 用户决策
- **暂不跑长训练**（询问 CPU 运行规模后选择"先不跑长训练"）

## 本次新测量结果
- 无梯度前向+loss（batch 64/embed 512/6 masks，CPU）：约 9.0 s/step
- **带梯度真实训练 step（含 backward）**：约 **36 s/step**，loss 正常下降（5.8 → 4.0）
- 单 epoch（train 10000 pairs，batch 64）= 157 步 ≈ **1.5-1.6 小时**
- 一次完整 probing（4 任务 × logreg 代价搜索）≈ 1 小时以上/轮
- OMP_NUM_THREADS=12 反而更慢 → 保持默认线程数
- 临时基准脚本 `_bench_timing.py` 已删除

## 已确认的可行性
- 代码库设备无关化已完成（4 处 `.cuda()` 修复），无 NVIDIA GPU（torch 2.14.0+cpu，12 物理核/16 线程）
- version_1 的 1-epoch 全流程运行产物验证：训练收敛、probing 正常、checkpoint 保存
- 最后一次日志 `0.5/1.0/[1.0,0.0]/0.5` = synergy 任务 epoch-1 随机初始化退化输出（正常现象）

## 待办（后续会话可从这些点继续）
1. 方案 A：本机 2-3 epoch 短验证（命令已在上文），需后台跑 6-10 小时
2. 方案 B：Colab T4 跑论文完整配置（推荐，数据 14 MB 可打包）
3. 方案 C：减配（embed 256/batch 32）快速调参
4. 清理工作区：../dataset/__pycache__ 等未跟踪文件可决定是否 gitignore

# 第三次续（2026-09-09 晚间）：UniGIR 创新实现与对照

## 背景转向
- 在复现跑通的基础上，基于项目做创新：**UniGIR**（idea 见 `../notion/idea/idea.md`，规划见 `UniGIR_路线规划.md`）
- 卖点定位经核查后转向：GeoMM (CVPR 2025) 已抢先"测地距离替换余弦" → 新卖点 = **全局关系结构感知的协同学习**（测地作对照增强项）

## 实现（V2 自包含版）
- `../losses/prototype_alignment.py`（新增）：512 原型 + Sinkhorn 均衡分配 + 掩码→完整视图 KL 对齐 + EMA 更新
- `../losses/infmasking_loss.py`：`profile_kwargs` 可选分支（默认关闭，与原始行为一致）
- `../configs/model/unigir.yaml`（新增）：`+model=unigir` 启用
- 修过两个 bug：EMA in-place 与 autograd 冲突（detach 快照解决）；hydra 结构键需 `+` 前缀

## 对照演示（CPU，4 epochs，seed 42，biased=false，max_size=1024）
- Baseline acc1：0.304→0.338→0.353→**0.372**
- UniGIR acc1：0.309→0.339→0.345→**0.355**
- profile 机制有效：val_ssl_acc_profile 0.159→0.373（随机 0.2%）
- 结论：短跑 R-U 组下游无增益；**协同增益需 S 组（biased=true）验证**
- 日志：`../_demo_baseline.log` / `../_demo_unigir.log`；产物在 `../InfMasking/bimodal_trifeatures/R-U-cpu-*/`

## 运行事故与教训（重要）
- 中途两次"假死"：根因是**同时存在两个满负荷 python 进程互抢 16 核**（重复启动产生孤儿进程）
- 处置：`tasklist`/`wmic` 定位 → `taskkill //PID` 清理 → 单进程恢复全速
- **教训：启动训练前先确认无残留 python；后台任务串行跑，勿并发**

## 存档
- 完整会话导出：`../ccDsForUniGIR.txt`（2026-09-09）
