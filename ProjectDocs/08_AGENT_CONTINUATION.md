# Agent 继续推进说明

## 2026-09-25 当前接续状态：G0 seed=7 UniGIR 正在运行

G3 MOSI 三 seed 已完成，UniGIR 的 acc@1 和 ROC AUC 在 3/3 个 seed 上高于 Baseline，平均差值为 +0.041 和 +0.037，满足进入 G0 的条件。G0 启动前已核对封存数据清单、脚本哈希和 Baseline/UniGIR 的 Hydra 实际解析。

当前远端任务：seed=42 Baseline 与 UniGIR 已分别以 `EXIT_CODE=0` 完成。原 seed=7 Baseline 于 23:48:57 以 `EXIT_CODE=1` 退出，stderr 报告 BatchNorm 更新处的 `CUDA error: invalid argument`。retry 任务 `InfMasking-G4-G0-s7-Baseline-Retry1-liangyl` 已从原 checkpoint 的 `epoch=2`、`global_step=471` 恢复，并于 01:20:13 以 `EXIT_CODE=0` 完成。最终 probing 为 share `0.959/0.999`、unique1 `0.806/0.974`、unique2 `0.778/0.969`、synergy `0.534/0.776`。原 seed=7 失败 checkpoint 不作为完成结果使用。

接续动作：retry checkpoint 已复制回本地，远端和本地 SHA256 均为 `C47E587F5A70CAFD534B88B5229B1661166B01930073A0F49244504E92F370A5`。seed=7 UniGIR 已于 07:05:05 进入 `Running`，使用 G0 封存数据、pair seed=42、model seed=7、10 epoch、queue=1024、linear probing 和独立运行目录。训练结束后先检查 `END`、退出码和 probing，再启动 seed=123 Baseline；所有任务仍须在单张 GTX 1080 Ti 上串行运行。

## 2026-09-24 当前接续状态：G3 MOSI 多 seed 完成，准备进入 G0

当前主线是：G1=Weak Go，G2=Uncertain，G3 MOSI 三个 model seed 的补充已经完成，G0 尚未训练。seed=42、7、123 上 UniGIR 的 acc@1 和 ROC AUC 均高于 Baseline，平均差值为 +0.041 和 +0.037，满足进入 G0 的条件。

正式协议见 `ProjectDocs/15_G4正式确认协议.md`，执行时以该文件为准：

1. MOSI：seed=7、123 已按 seed=42 的 10 epoch、batch size 32、`by_fit` probing 完成；MOSI 不使用 Trifeatures 的 pair sampling seed。
2. G0：现在可以启动；使用 `trifeatures_g0_seed20260924`，pair sampling seed 固定为 42，model seed 为 42、7、123，首轮 10 epoch。
3. G0 的 Baseline 和 UniGIR 各自使用独立目录、日志和 checkpoint；test 指标在全部训练完成后统一读取。
4. 当前不启动 100 epoch，不增加新的 Trifeatures 机制变体，不用 G0 结果调参。

本轮已生成 `outputs/g4_mosi_seed_manifest_20260924/manifest.txt`，核对了 G3 运行所需代码、配置、MOSI 数据哈希和远端环境；新增的 MOSI 计划任务脚本已同步到 winpc。G0 启动前仍需针对 `DataRoot` 和 `PairSeed` 重新检查实际解析配置。

## 2026-09-24 当前接续状态：G3 MOSI 10 epoch pilot 完成

`main_multibench.py` 已补齐 MultiBench 的 checkpoint 保存和自动恢复。默认每个 epoch 保存当前实验目录下的 `checkpoints\last.ckpt`；显式 `resume_ckpt_path` 优先，没有显式路径时自动查找该文件；checkpoint 回调不依赖 linear probing。相关配置字段已加入 `configs/train_multibench.yaml` 和 `configs/train_multibench_all-mod.yaml`。

本地语法检查、winpc 端相关 11 项测试和默认/MOSI Hydra 配置解析均已通过。Baseline 和 UniGIR 各自先运行 1 epoch，再以相同实验名运行 2 epoch；两次恢复均退出码为 0，并在 stdout 中记录了 `Resume checkpoint` 路径。

G3 MOSI 10 epoch 配对 pilot 已完成。两组使用 seed=42、batch size 32、GPU 0、10 epoch、`deterministic=true`、`num_workers=0`、`linear_probing.frequency=by_fit`。

| 方法 | acc@1 | ROC AUC | 结果目录 |
|---|---:|---:|---|
| Baseline | 0.573 | 0.655 | `outputs/winpc_20260924_g3_mosi_10ep_seed42_baseline_v2/` |
| UniGIR | 0.641 | 0.704 | `outputs/winpc_20260924_g3_mosi_10ep_seed42_unigir_v2/` |
| UniGIR - Baseline | +0.068 | +0.049 | — |

两组均为 `EXIT_CODE=0`，checkpoint 已复制回本地并完成 SHA256 核对；winpc GPU 0 已空闲，没有残留 Python 训练进程。G3 只给出单 seed、10 epoch 的初步迁移证据，不能直接写成稳定性能结论。

当前主线：G1=Weak Go，G2=Uncertain，G3=正向初步证据。下一步整理 G4 正式确认协议，固定代码、数据版本、seed、训练预算和最终 probing；不启动 100 epoch，不使用 G0 反复调参，不增加新的 Trifeatures 机制变体。

阶段汇报规则：`ProjectDocs/导师阶段汇报.md` 以 2026-09-12 为上次汇报节点，只记录之后尚未确认汇报过的内容。用户以后给出新的已汇报日期后，先按日期清理该文件，再继续追加新的节点；`ProjectDocs/05_导师汇报.md` 继续作为当前完整状态文档。

## 2026-09-24 当前接续状态：G3 MOSI loader 与 1 epoch smoke 完成

阶段汇报规则：`ProjectDocs/导师阶段汇报.md` 以 2026-09-12 为上次汇报节点，只记录之后尚未确认汇报过的内容。用户以后给出新的已汇报日期后，先按日期清理该文件，再继续追加新的节点；`ProjectDocs/05_导师汇报.md` 继续作为当前完整状态文档。

G2 的 C1 shuffled profile 和 C2 queue=0 均已在 winpc GPU 0 上完成。两项任务都使用 development split、S 组、seed=42、10 epoch、1024 pair、batch size 64、四项 probing；G1 正常 UniGIR 使用 queue=1024、$K=128$、$\alpha=0.25$、单向 profile。

| 对照 | share acc@1/AUC | unique1 acc@1/AUC | unique2 acc@1/AUC | synergy acc@1/AUC | 平均 acc@1/AUC |
|---|---|---|---|---|---|
| 正常 UniGIR，queue=1024 | 0.520/0.874 | 0.395/0.829 | 0.309/0.753 | 0.519/0.595 | 0.436/0.763 |
| C1 shuffled profile | 0.540/0.903 | 0.326/0.758 | 0.311/0.754 | 0.550/0.550 | 0.432/0.741 |
| C2 queue=0 | 0.475/0.861 | 0.360/0.798 | 0.264/0.731 | 0.546/0.645 | 0.411/0.759 |

C1 相对正常配置使 unique1 ROC AUC 下降 0.071、synergy ROC AUC 下降 0.045，说明样本对应关系对部分下游信号有贡献；C2 使整体平均 acc@1 下降 0.025，但 synergy ROC AUC 上升 0.050，说明 queue 对整体表示有帮助，却不是 synergy 的必要条件。两项都只有一个 seed，最终日志没有 profile accuracy、profile loss、使用熵和 active prototypes，G2 当前记为 Uncertain。

当前不要启动更多 Trifeatures 机制变体、100 epoch 长训练或 G0 未见数据训练。MOSI 的数据、loader、Baseline 和 UniGIR 的 1 epoch smoke 已完成，下一步先整理 G3 结果，再决定是否运行 10 epoch 的配对 pilot。MOSI 没有直接 synergy 标签，结果只能用于真实任务迁移判断；1 epoch 差值不构成性能结论。

MOSI 预检已进入 loader 阶段：`dataset/catalog.json` 和 `dataset/affect/get_data.py` 的入口存在，远端 Hydra 配置解析通过。轻薄本已按内置 Google Drive ID 下载文件并传到 `C:\Users\liangyelian\work\InfMasking_desktop_20260923\dataset\data\mosi\mosi_data.pkl`；文件大小为 154041300 字节，SHA256 为 `1a113ad5edc8b9b625a7e29e6b94a97ecade70494c4e022d9f7ba5affca3bbdc`，远端哈希一致。

远端 `Affect` 读取结果为 train/val/test=1284/229/686，标签计数分别为 `{0: 605, 1: 679}`、`{0: 105, 1: 124}`、`{0: 409, 1: 277}`。MultiBench SSL loader 的两个视图均包含 vision/text 张量，形状为 `(4, 30, 20)` 和 `(4, 30, 300)`；监督 loader 返回 `(4,)` 的二分类标签。seed=42、batch size 32、GPU 0、1 epoch 的 UniGIR smoke 退出码为 0，acc@1/ROC AUC 为 0.545/0.631；同设置 Baseline smoke 退出码为 0，acc@1/ROC AUC 为 0.553/0.578。两个远端实验均生成独立 checkpoint，UniGIR checkpoint 已复制到 `outputs/winpc_20260924_g3_mosi_unigir_smoke_v2/` 并完成两端 SHA256 核对。winpc 的 Google Drive 直连问题不再阻塞后续工作，因为数据已由轻薄本传输到位。

G3 执行中修复了两处问题：`models/transformer.py` 将 `sentence_transformers` 改为 `LanguageEncoder` 使用时导入；`evaluation/linear_probe.py` 将确定性训练下的多分类 ROC AUC 放到 CPU 计算，避开 CUDA 的确定性 cumsum 内核错误。修复后 Baseline/UniGIR smoke 均通过。

## 2026-09-24 当前接续状态：G2 C1 完成，准备 G2 C2

目标主机仍为 `winpc`，远程项目目录为 `C:\Users\liangyelian\work\InfMasking_desktop_20260923`。当前只看到一张 NVIDIA GeForce GTX 1080 Ti，实验继续在 GPU 0 串行运行。G0 新数据 `dataset/data/trifeatures_g0_seed20260924` 已封存，不能用于当前方法选择。

G2 C1 已完成。正式设置为 development split、S 组、seed=42、`biased=true`、`max_size=1024`、batch size 64、10 epoch、$K=128$、queue=1024、$\alpha=0.25$、单向 profile、四项 probing，并启用 `shuffle_targets=true`。任务 `InfMasking-Winpc-G2-C1-seed42-Shuffled` 于 2026-09-24 14:19:13 以 `EXIT_CODE=0` 完成，日志位于 `outputs/winpc_20260924_g2_c1_seed42_shuffled/`，远端 checkpoint 已生成。

| 设置 | share acc@1/AUC | unique1 acc@1/AUC | unique2 acc@1/AUC | synergy acc@1/AUC | 平均 acc@1/AUC |
|---|---|---|---|---|---|
| 正常 UniGIR，queue=1024 | 0.520/0.874 | 0.395/0.829 | 0.309/0.753 | 0.519/0.595 | 0.436/0.763 |
| shuffled profile | 0.540/0.903 | 0.326/0.758 | 0.311/0.754 | 0.550/0.550 | 0.432/0.741 |
| shuffled - 正常 | +0.020/+0.029 | -0.069/-0.071 | +0.002/+0.001 | +0.031/-0.045 | -0.004/-0.022 |

C1 说明样本对应关系对 unique1 和 synergy ROC AUC 的部分下游信号有贡献，但 share 和 synergy acc@1 上升，且本次日志没有 profile accuracy，因此不能写成完整机制证明。当前状态仍为 Weak Go，下一步运行 C2 queue=0。

为 C2 已完成的代码准备：`run_scripts/winpc_start_experiment.ps1` 和 `run_scripts/winpc_schedule_experiment.ps1` 都新增 `[int]$QueueSize=1024`，并把它写入 Hydra override 和 status 日志。queue=0 的 1 epoch、64 pair smoke 已以 `EXIT_CODE=0` 完成并生成独立 checkpoint。

接续动作：

1. 只读确认 GPU 0 空闲、没有其他训练进程；
2. 注册 `InfMasking-Winpc-G2-C2-seed42-Queue0`，设置 `-Method unigir -Seed 42 -GpuIndex 0 -MaxEpochs 10 -MaxSize 1024 -QueueSize 0 -EnableLinearProbe`；
3. 任务结束后核对 `END`、exit code、四项 probing、stderr 和 `last.ckpt`，把日志复制到 `outputs/winpc_20260924_g2_c2_seed42_queue0/`；
4. 将 queue=0 与 G1 seed=42 的 queue=1024 结果逐项比较，再决定是否进入 MOSI 离线预检或停止扩大 UniGIR。

不要停止用户任务，不要删除监控任务，不要使用 G0 新数据调参，不要启动 100 epoch。C2 只改变 queue size，`shuffle_targets` 必须保持 false。

## 2026-09-24 当前接续状态：G1 三 seed 完成，达到 Weak Go

最近一次任务已经完成 G0 数据协议冻结和 G1 三 seed 配对 pilot。目标主机是 `winpc`，远程项目目录为 `C:\Users\liangyelian\work\InfMasking_desktop_20260923`；实际只看到一张 NVIDIA GeForce GTX 1080 Ti，因此实验按 GPU 0 串行运行。

G0 状态：新数据根目录为 `dataset/data/trifeatures_g0_seed20260924`，数据生成 seed=`20260924`，train/test=2400/200，逐文件 SHA256、摘要、生成日志和关键代码哈希已保存在本地 `outputs/g0_protocol_seed20260924/` 和 `ProjectDocs/14_G0协议与版本清单.md`。新数据暂不参与方法选择。

G1 固定设置：当前 development split、S 组、`biased=true`、`max_size=1024`、batch size 64、embed_dim=256、`num_mask=3`、10 epoch、`num_workers=0`、训练结束后一次 probing。UniGIR 为单向 profile、$K=128$、queue=1024、$\alpha=0.25$、`cross=false`。seed=42、7、123 的 Baseline 和 UniGIR 均退出码为 0，并生成独立 `last.ckpt`。

| seed | synergy acc@1 变化 | synergy ROC AUC 变化 | 四任务平均 acc@1 变化 | 四任务平均 ROC AUC 变化 |
|---:|---:|---:|---:|---:|
| 42 | +0.006 | +0.001 | +0.032 | +0.014 |
| 7 | +0.010 | +0.022 | -0.011 | -0.002 |
| 123 | +0.009 | -0.016 | +0.022 | +0.004 |
| 三 seed 平均 | +0.008 | +0.002 | +0.014 | +0.005 |

判断：2/3 个 seed 的 synergy ROC AUC 提升，三 seed 平均变化为正，且四任务平均 acc@1 没有下降，达到预设 Weak Go；seed=123 的 synergy ROC AUC 下降 0.016，平均提升约 0.002，因此未达到 Strong Go。当前不能进入 100 epoch 或正式未见数据结果。

接续动作：先在 development split、seed=42 上做 shuffled profile 对照，再做 queue=0 对照。优先核对 `main_trifeatures.py`、`losses/prototype_alignment.py`、`losses/infmasking_loss.py` 和 `ProjectDocs/02_实验计划.md` 的 G2 定义；实验仍使用独立任务、日志、checkpoint 和断点续跑设置。若机制对照不能支持核心假设，停止扩大 UniGIR；若支持，再复查 seed=7、123 并进行 MOSI 离线预检。

当前本地结果日志：`outputs/winpc_20260924_s123_baseline/`、`outputs/winpc_20260924_s123_unigir/`；远端监控任务 `InfMasking-Winpc-Monitor-liangyl` 保留。不要删除用户任务，不要把 G0 数据提前用于调参。

## 2026-09-24 当前接续状态：seed=7 配对实验完成

最近一次任务完成了训练入口的断点续跑修复，并在 winpc 上启动了原计划的下一组实验。目标主机是 `winpc`，远程项目目录为 `C:\Users\liangyelian\work\InfMasking_desktop_20260923`。当前 `nvidia-smi` 只看到一张 NVIDIA GeForce GTX 1080 Ti，GPU 0 没有第二张卡记录，因此不能按两卡并行的前提继续安排实验。

代码状态：`main_trifeatures.py` 现在为每个实验建立稳定的实验根目录、日志目录和 `checkpoints` 目录；训练模式始终注册 `ModelCheckpoint`，每个 epoch 保存 `last.ckpt`；启动时自动检测 `last.ckpt`，并将路径传给 `trainer.fit(..., ckpt_path=...)`。配置新增 `checkpoint_dir`、`resume_ckpt_path` 和 `logger_version`。

验证状态：远程 11 项单元测试全部通过。低成本 smoke 第一次运行生成 `last.ckpt`，第二次使用同一实验名启动时显示从 `last.ckpt` 恢复，并从 1 epoch 运行到 2 epoch。这个结果确认恢复路径实际生效。

运行状态：seed=7 Baseline 和 UniGIR 均使用 `GPU 0`、`max_size=1024`、10 epoch、`num_workers=0`、embed_dim=256、num_mask=3、开启四项 probing，分别使用实验名 `winpc-S-seed7-baseline` 和 `winpc-S-seed7-unigir`，退出码均为 0。Baseline 的 share、unique1、unique2、synergy acc@1/AUC 为 0.476/0.856、0.518/0.879、0.428/0.833、0.549/0.701，平均 acc@1/ROC AUC 为 0.493/0.817；UniGIR 为 0.427/0.827、0.548/0.893、0.392/0.818、0.559/0.723，平均为 0.482/0.816。UniGIR 配置为单向 profile、$K=128$、queue=1024、$\alpha=0.25$。两个 checkpoint 已分别保存，监控任务 `InfMasking-Winpc-Monitor-liangyl` 已记录完成状态。

G0 已完成。winpc 上已生成独立数据根目录 `dataset/data/trifeatures_g0_seed20260924`，数据生成 seed 为 `20260924`，train/test=2400/200；逐文件哈希、摘要、生成日志和协议清单已复制到本地 `outputs/g0_protocol_seed20260924/`。新数据暂不用于方法选择。

接续动作：

1. 读取本地 `outputs/winpc_20260924_s7_baseline` 和 `outputs/winpc_20260924_s7_unigir`，将 seed=7 差值与 seed=42 并列分析；
2. 根据 G1 条件决定是否在 development split 启动 seed=123，启动前仍使用独立实验目录、checkpoint 和脱离 SSH 的任务计划；
3. 若 `nvidia-smi` 后续显示第二张可用卡，再把两个互不相同的实验分配到 GPU 0 和 GPU 1；当前不要为了满足两卡计划而抢占或伪造 GPU 1；
4. 新 G0 数据只在方法冻结后用于正式确认；不重复启动 seed=42 或 seed=7，不直接进入 100 epoch，不删除用户已有任务；只清理本任务创建且已完成的计划任务。

当前监控日志：`C:\Users\liangyelian\work\InfMasking_desktop_20260923\winpc_monitor\monitor.log`。

---

## 2026-09-24 当前接续状态：winpc seed=42 配对 pilot 已完成

最近一次任务已完成 Windows 台式机上的修复版 S 组配对 pilot。目标主机是 `winpc`，不是 `lab-gpu`；硬件为 NVIDIA GeForce GTX 1080 Ti，显存 11264 MiB。远程项目目录为 `C:\Users\liangyelian\work\InfMasking_desktop_20260923`，本地结果日志位于 [`outputs/winpc_20260924_s42`](../outputs/winpc_20260924_s42)。

环境已经验证：Python 3.11.15、PyTorch `2.1.0+cu118`、CUDA 可用、PyTorch Lightning 2.1.1、TorchMetrics 1.2.0、TensorBoard 2.14.1、timm 0.9.12、NumPy 1.26.4。WSL 没有发行版，下载 Ubuntu 时因 GitHub 请求超时，当前路线使用 Windows 原生 Python；不要把这个环境写成 Linux 或 `lab-gpu` 环境。

本次设置固定为 S 组、`biased=true`、seed=42、`max_size=1024`、batch size 64、`num_workers=0`、10 epoch、embed_dim=256、num_mask=3、mask ratio=0.7、固定 epoch 后一次 probing。UniGIR 为单向 profile、$K=128$、queue=1024、$\alpha=0.25$、`cross=false`。

| 方法 | share acc@1 | share ROC AUC | unique1 acc@1 | unique1 ROC AUC | unique2 acc@1 | unique2 ROC AUC | synergy acc@1 | synergy ROC AUC | 平均 acc@1 | 平均 ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.463 | 0.859 | 0.319 | 0.796 | 0.322 | 0.747 | 0.513 | 0.594 | 0.404 | 0.749 |
| UniGIR | 0.520 | 0.874 | 0.395 | 0.829 | 0.309 | 0.753 | 0.519 | 0.595 | 0.436 | 0.763 |
| UniGIR - Baseline | +0.057 | +0.015 | +0.076 | +0.033 | -0.013 | +0.006 | +0.006 | +0.001 | +0.032 | +0.014 |

当时判断：这是一条修复版单 seed 的开发信号。share、unique1 和四任务均值上升，但 synergy 只变化 +0.006 acc@1、+0.001 ROC AUC，不能写成 synergy 已经稳定改善。随后已补跑 seed=7、123；当前总状态以本文最上方的 G1 三 seed 结论为准。Baseline 日志没有 `EXIT_CODE` 行，但包含 `Trainer.fit stopped` 和四项 probing，Python 已退出；UniGIR 定时任务返回码为 0。

## 当时的下一步行动

1. 删除已完成的 UniGIR 任务计划，保留本地日志和运行脚本；
2. 将本次结果作为开发 pilot，按相同设置运行 seed=7、123，先判断 G1 的三 seed 条件；
3. 三 seed 已达到 Weak Go，当前转入 shuffled profile 和 queue=0 机制对照；不直接启动 100 epoch 或正式未见数据实验。

接续时先读取 `progress.md` 顶部、本文本节、`ProjectDocs/03_实验记录.md` 的 2026-09-24 小节和 `ProjectDocs/04_结果分析.md` 的 2026-09-24 小节；先检查 `winpc` 的 GPU 和 Python 进程，不重复启动 seed=42。

更新时间：2026-09-23

## 两个远程目标必须分开

项目当前有两个完全不同的计算目标，后续任务不能混用它们的事实、路径、账户、硬件或命令。

| 目标名称 | 已核实信息 | 当前边界 |
|---|---|---|
| `lab-gpu` | 实验室共用 Linux 服务器 `seclab03`，地址 `121.48.227.136`，账户 `liangyl`；只读检查得到 Ubuntu 20.04.5 和 NVIDIA GeForce RTX 4090，GPU 当时已有 Python 进程占用 | 共享机器。发现他人任务时只记录并等待，不停止进程、不启动容器、不上传代码。本文中关于 Linux、Docker 和 4090 的旧记录只适用于它。 |
| `desktop-wsl` | 用户的新 Windows 台式机，地址 `172.16.1.113`，账户 `liangyelian`；用户确认已安装 WSL | 硬件、GPU、WSL 发行版、WSL 版本、Docker、磁盘、默认 SSH shell 和目标路径都还没有读取。不能把它当作 4090 机器，也不能假定 Docker 或 GPU 可用。 |

后续判断规则：看到 `lab-gpu` 才能使用实验室服务器的已核实事实；看到 `desktop-wsl` 必须先做 Windows/WSL 只读检查。两台机器都不能使用对方的路径和用户。台式机部署按 `ProjectDocs/13_台式机Windows与WSL部署计划.md` 继续，实验室服务器按 `ProjectDocs/12_服务器SSH与Docker运行计划.md` 继续。

## `desktop-wsl` 当前连接结果

本机到 `172.16.1.113` 的 Ping 成功，TCP 22 可达；使用 `liangyelian@172.16.1.113` 做非交互 SSH 身份检查时返回 `Permission denied (publickey,password,keyboard-interactive)`。这说明当前阻塞在认证，尚未读取台式机的主机名、Windows/WSL、Python、Docker、GPU 或磁盘状态。

前一版建议错误地引用了实验室 `lab-gpu` 专用密钥 `id_ed25519_lab`。台式机必须使用独立的 `id_ed25519_desktop_wsl` 密钥：用户在本机生成密钥，只把对应的 `.pub` 公钥添加到台式机账户；现有 `id_ed25519_lab` 不复制、不修改、不复用。不要把密码写入命令、脚本、项目或工具输入，不要为了绕过认证关闭主机校验。

当前本机独立公钥指纹为 `SHA256:Ab4NcETaORy3jj9JEQdcQOffjKXByjpYJu6addIBi6E`。SSH 调试日志确认该公钥已发送到台式机，但远端拒绝，因此下一步核对远端 `$env:USERNAME`、`$env:USERPROFILE`、`authorized_keys` 文件名和指纹；如果账户属于 Windows Administrators 组，还要检查 OpenSSH 的管理员公钥文件。

## 开始任务前

1. 先读取项目根目录的 `progress.md`；
2. 再读取 `CLAUDE.md`、`ProjectDocs/00_研究总览.md`、`ProjectDocs/me/思考.md` 和本文档；
3. 在 `10_任务启动记录.md` 写明任务、目的、方法、预期结果和预计耗时；
4. 读取 `11_架构与依赖说明.md`，确认本次任务涉及的入口和模块边界；
5. 在聊天窗口输出同样的信息；
6. 检查是否存在 Python 训练进程，不要重复启动实验。

## 服务器连接当前状态

已核对 [slepp/ssh-mcp](https://github.com/slepp/ssh-mcp)：它通过本机 OpenSSH 配置、密钥、SSH Agent 和 ProxyJump 连接远端，并提供 `ssh_exec`、会话、文件查看和传输工具。当前 Codex 会话没有动态加载该 MCP，但已在临时 Python 3.13 环境中验证服务可以启动，MCP 握手返回版本 0.2.0 和 17 个工具。

本机 OpenSSH 可用，`C:\Users\breeze\.ssh` 没有 `config` 文件，`known_hosts` 只记录 `github.com`。当前服务器目标已由用户提供为 `liangyl@121.48.227.136:22`，但第一次严格校验因缺少已确认主机密钥而停止。公开扫描得到的 ED25519 指纹为 `SHA256:WqCfqbGXtModqG/8GKax7+I6z1lS14nqtlGRCFkR9SY`，必须先与实验室记录核对，后续 Agent 不得直接接受未知指纹或关闭校验。核对通过后，先执行只读预检：`hostname`、`uname -a`、`id`、`nvidia-smi`；只有用户明确要求继续且不涉及状态变化时，才读取 `docker --version`、`docker info`、`df -h`、`df -i`、`findmnt`。不要输入密码或复制私钥，不执行 `sudo`、`docker run`、`docker pull`、`docker build`、数据同步或训练。

当前认证尝试结果：`ssh-mcp` 使用本机默认 SSH 私钥并启用 `BatchMode=yes` 后返回 `Permission denied (publickey,password)`。后续 Agent 不得反复尝试密码，不得索取或记录密码；等待用户在本机配置正确的 SSH Agent/私钥或手动完成登录后，再执行只读预检。

推荐的接续方式：用户生成专用 `id_ed25519_lab`，通过已有 MobaXterm 登录把 `.pub` 公钥加入服务器账户的 `~/.ssh/authorized_keys`，再在 `C:\Users\breeze\.ssh\config` 中创建 `lab-gpu` 别名。用户先在本机运行 `ssh -o BatchMode=yes lab-gpu "hostname; id"` 做只读验证，成功后 Agent 才能继续 `ssh-mcp` 预检。不要覆盖现有 `id_rsa`，不要把私钥或 passphrase 放入项目、Docker、聊天或文档。

项目已提供 `run_scripts/setup_ssh_access.ps1` 辅助完成本机部分。后续 Agent 应先让用户运行 `-WriteConfig` 版本生成密钥和配置，再由用户通过 MobaXterm 添加公钥，最后运行 `-TestConnection`。Agent 不代为生成私钥，不读取私钥内容，不把密码或 passphrase 写入命令和文档。

用户已确认 `ssh lab-gpu` 可以交互式登录，但 `ssh-mcp` 在 BatchMode 下认证失败，原因是私钥 passphrase 尚未加载到本机 SSH Agent。后续 Agent 应让用户在本机运行 `Start-Service ssh-agent` 和 `ssh-add "$env:USERPROFILE\.ssh\id_ed25519_lab"`，用户自行输入 passphrase；不要通过聊天、远端命令或 MCP 传输 passphrase。只有 `ssh -o BatchMode=yes lab-gpu "hostname; id"` 成功后，才继续远端 GPU 检查。

当前已完成一次只读连接：`ssh-mcp` 成功进入 `seclab03`，账户 `liangyl` 属于 `docker` 组；GPU 0 为 RTX 4090，显存使用 5360/24564 MiB，利用率 19%，有 Python 进程占用约 5358 MiB。GPU 正在使用，后续 Agent 必须停止远端操作，不得查看他人进程详情、停止进程、启动 Docker、同步代码或训练。待 GPU 空闲并符合课题组规则后，才补充 Docker、磁盘和挂载的只读检查。

## 当前研究主线

项目基于 InfMasking，验证完整视图到原型的关系轮廓能否帮助掩码视图保留 synergy 信息。当前方法为 UniGIR V2：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{InfMasking}}
+
\alpha\mathcal{L}_{\mathrm{profile}}.
$$

当前候选固定为单向预测、$K=128$、queue=1024、$\alpha=0.25$。项目目标是尽快形成有充分证据的 CCF-A 主会候选论文，完成 UniGIR 本身不是继续投入的理由。当前 UniGIR 状态为 Weak Go，下一步是 G2 机制对照。

后续顺序固定为：G0 协议冻结、G1 三 seed 开发 pilot、G2 shuffled profile 与 queue=0、G3 MOSI、G4 正式证据。不要在 G1 前增加 cross 或测地距离，也不要在 G2、G3 前启动完整 100 epoch 合成长训练。

## 2026-09-19 必须知道的复核结果

旧版实现和评测协议存在四个问题：

1. `PrototypeAlignment._update` 名为 EMA，实际不是标准 EMA；
2. 原型随机初始化消耗全局 RNG，使 Baseline 和 UniGIR 的同 seed 对照不完全配对；
3. 训练入口没有统一设置 Python `random` 和 worker seed；
4. 每个 epoch 的 probing 使用官方 test split，不能用它 early stopping 或挑选 checkpoint。

已经完成的代码修复：

- `losses/prototype_alignment.py` 使用 assignment-weighted centroid 做 EMA；
- 原型通过独立的 CPU generator 初始化，`init_seed=${seed}`；
- `main_trifeatures.py` 使用 `seed_everything(seed, workers=True)`；
- 默认 `probe_frequency=by_fit`，固定训练长度，官方测试只在训练结束后评测一次；
- 默认 `checkpoint_monitor=null`、`enable_early_stopping=false`，只保存 `last.ckpt`；
- `evaluation/linear_probe.py` 将最终指标以 `final_*` 写入 logger；
- 新增 target entropy、target confidence、prediction usage entropy 和 prototype mean absolute cosine；
- UniGIR 默认 $K$ 改为 128；
- Trifeatures 默认单进程、单设备、确定性训练、`num_workers=0`；
- 新增 `.gitignore` 和 `run_scripts/gpu_pilot_s_group.ps1`。

## 验证证据

运行命令：

```powershell
.venv\Scripts\python.exe -m unittest tests.test_prototype_alignment -v
```

原有五个核心测试全部通过；本轮又增加了 MOSI 下载目录测试，当前共六个测试通过：

- 原型初始化不推进全局 RNG；
- EMA 结果符合预期公式；
- eval 不更新原型和 queue；
- train forward 更新 queue 且 loss 可以反向传播。
- `by_fit` 不在每个 validation epoch probing，并在 fit 结束后写入 `final_*` 指标。

修复后端到端冒烟也已通过：S 组、seed=123、max_size=64、embed_dim=64、num_mask=1、1 epoch、关闭 probing。日志和 TensorBoard 位于：

```text
InfMasking/bimodal_trifeatures/postfix-smoke/
```

新增指标均为有限值。冒烟只检查运行，不用于性能判断。

## 历史结果如何使用

修复前的两个 seed 结果保留在 `03_实验记录.md` 和 `04_结果分析.md`：

| seed | 方法 | synergy acc@1 | synergy ROC AUC | 平均 acc@1 |
|---:|---|---:|---:|---:|
| 42 | Baseline | 0.501 | 0.587 | 0.350 |
| 42 | UniGIR | 0.543 | 0.621 | 0.355 |
| 7 | Baseline | 0.520 | 0.576 | 0.375 |
| 7 | UniGIR | 0.533 | 0.581 | 0.372 |

这些结果只能称为预修复探索结果。不要将它们与修复后的实验直接合并，不要继续引用它们证明当前代码有效。

旧版 target usage entropy 和 active prototypes 受 Sinkhorn 均衡约束直接影响，不能单独证明没有塌缩。后续必须同时查看：

- `profile_prediction_usage_entropy`；
- `profile_target_confidence`；
- `profile_prototype_mean_abs_cosine`；
- `ssl_acc_profile`；
- 最终下游指标。

## 下一步唯一优先行动

等待 GPU 期间先完成 G0：

1. 使用已加入的 `data.data_module.data_root` 指定独立 Trifeatures 根目录和数据 seed，确保新数据不覆盖当前开发数据；
2. 生成实验版本清单，记录 Git 状态、关键文件 SHA256、配置、数据根目录、数据 seed 和模型 seed；
3. 对 MOSI 做离线代码预检。当前仓库有配置和自动下载入口，本地缺少 `dataset/data/mosi/mosi_data.pkl`；
4. 不下载数据时，也可以先检查 `main_multibench.py` 与 UniGIR 配置的兼容点。

当前机器没有 CUDA。G0 完成并获得 GPU 后运行：

```powershell
powershell -ExecutionPolicy Bypass -File run_scripts/gpu_pilot_s_group.ps1
```

脚本先按 seed=42、7、123，依次运行 Baseline 和 UniGIR。每组固定 10 epoch，训练结束后才执行一次 probing。脚本检测不到 CUDA 时会立即停止，不会退回 CPU 长训练；已完成且日志含 `EXIT_CODE=0` 的组会自动跳过。

先只等待 seed=42 两组完成并检查结果。若训练稳定，再继续剩余 seed。pilot 至少 Weak Go 后先做 shuffled profile 和 queue=0；机制成立后再运行 MOSI；不要直接启动 100 epoch 主实验。

## 数据与评测协议

- 开发数据：当前 S 组，`biased=true`；
- pilot：max_size=1024、embed_dim=256、num_mask=3、10 epoch；
- seeds：42、7、123；
- 比较：InfMasking Baseline 与 UniGIR 单向 queue=1024；
- 固定 epoch，不 early stop；
- 官方测试集只在 fit 结束后评测一次；
- 第一报告指标为 synergy ROC AUC，但不使用测试指标做 checkpoint 选择；
- 保存最后 checkpoint；
- 记录每组运行时间、GPU 型号、显存、PyTorch 和 CUDA 版本。

当前 Trifeatures test split 已参与开发选择，不能作为正式未见测试。方法冻结后，使用未见数据 seed 生成新的完整 train/test 数据，在新 train 上从头训练两种方法，再评测对应 test。模型 seed 和数据 seed 分开记录。

G1 状态规则：

- Strong Go：3/3 seed 的 synergy ROC AUC 提升，平均绝对提升至少 0.01；
- Weak Go：至少 2/3 seed 提升，三 seed 平均变化为正；
- 两种 Go 均要求四任务平均 acc@1 的平均下降不超过 0.01，诊断量无明显退化；
- 其他情况记为 Uncertain 或 No-Go，不自动扩大实验。

机制对照中，shuffled profile 优先级最高。`queue=0` 已等价于当前代码的 batch-only profile，不要重复创建两个实验。

## 运行和磁盘约束

- 当前环境：Python 3.13.0、PyTorch 2.14.0+cpu；
- 当前机器：16 逻辑处理器、约 15.7 GB 可用内存、无 NVIDIA GPU；
- D 盘剩余约 68.8 GB；
- `InfMasking/` 下有 22 个 checkpoint，约 2.19 GB；
- checkpoint 和日志已被 `.gitignore` 排除，但没有删除；
- 工作区包含大量历史改动，不得使用 `git reset --hard` 或覆盖用户文件；
- 每次只运行一个训练进程。

`max_size` 是 pair 数，不是基础图片数。S 组 `max_size=1e4` 对应 10000 个训练 pair 和 404 个测试 pair；batch size 64 时每个 epoch 约 157 个训练 step。不要再把主实验写成只有 2400 个训练样本。

当前候选配置已经参考过旧 test split。修复版 pilot 仍是开发实验；只增加模型 seed 不能消除重复查看 test split 的选择偏差。

MOSI 有两模态与三模态配置，第一轮优先复现 `run_scripts/multibench_run.sh` 的 vision、text 两模态设置。MOSI 是情感分类任务，结果只能支持真实任务迁移，不能单独证明 synergy 机制。

2026-09-21 的代码复审已确认：6 个单元测试通过；关键模块语法检查通过；Trifeatures 的独立根目录配置和 MOSI 下载目录保护已通过配置与单元测试；MultiBench 的 MOSI + UniGIR Hydra 配置解析通过。当前没有 CUDA，不能启动 G1 pilot。

## Linux SSH 与 Docker 服务器接续规则

本机 Windows 的 PowerShell GPU 脚本不能直接搬到服务器。服务器使用 `ProjectDocs/12_服务器SSH与Docker运行计划.md` 中的 Linux 流程：先在宿主机检查 GPU 和 Docker，再启动指定 GPU 的容器；代码、数据、runs 和 cache 分别挂载；训练放在 `tmux` 或 `screen` 中；每次只启动一个 seed、一个方法和一个 GPU。

数据挂载目标使用 `/workspace/InfMasking/dataset/data`，因为 `dataset/catalog.json` 中的 MOSI 和其他 MultiBench 路径是相对路径；不要只挂载到 `/workspace/data` 后直接运行。

服务器环境优先对齐原始 InfMasking 的 Linux/Conda 基线：Python 3.8.18、PyTorch 2.1.0、PyTorch CUDA 11.8、torchvision 0.16.0、PyTorch Lightning 2.1.1。Windows uv 环境不作为服务器基线。官方脚本中的多 seed 循环只能参考，实际运行必须拆成单 seed、单方法、单 GPU。

服务器执行顺序固定为：镜像 CUDA 预检 → 容器内单元测试 → 1 epoch GPU 冒烟 → G0 数据与版本清单 → seed=42 Baseline → seed=42 UniGIR → 检查结果和资源成本 → 再决定是否运行 seed=7、123。GPU 仍属于共用资源，不能结束或暂停他人进程，也不能把 GPU 空闲状态当成长期保证。

服务器运行前必须记录镜像 digest、GPU 型号、显存、驱动、PyTorch、CUDA、代码哈希、数据哈希、完整配置、运行命令和退出码。没有这些记录的结果只能作为临时开发结果。

## 完成任务后

1. 更新 `progress.md` 的状态、证据和下一步；
2. 更新 `10_任务启动记录.md`；
3. 状态变化同步到 `00_研究总览.md`；
4. 实验事实追加到 `03_实验记录.md`，解释写入 `04_结果分析.md`；
5. 方法变化更新 `01_研究想法与方法.md` 和 `07_动机与行动记录.md`；
6. 更新本文档的日期和下一步唯一行动；
7. 运行单元测试、脚本语法检查和 `git diff --check`。
