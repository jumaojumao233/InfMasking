# UniGIR 创新路线规划（2026-09-09）

> 关联文档：`../notion/idea/idea.md`（UniGIR 原始 idea）、`运行 InfMasking 项目.md`（基线复现记录）
> 目标：ACM Multimedia 2027（Thematic Area: Multimodal Fusion）

---

## 一、定位警示（先读这一节）

原始 idea 的叙事是**"首次将测地距离引入多模态对比学习"**。经论文事实核查，**该说法已被 GeoMM (CVPR 2025) 抢先**：

- **GeoMM (CVPR 2025)**：已经提出"用测地距离替换余弦距离"，做法是 **k-NN 图 + Floyd 最短路 + 分层 K-means（在聚类中心上跑 Floyd）实现可计算 + 队列增量更新**。动机（"余弦的球面假设与流形数据失配"）与 UniGIR 几乎逐句相同。
- **Indra (NeurIPS 2025)**：工作区已有完整代码（`../Indra`），是 VLM/ALM 评测下的 **Yoneda 嵌入**方法，O(n²) 仅推理用。
- **InfMasking (NeurIPS 2025 Spotlight)**：本地已复现跑通。

**剩余差异化空间的客观评估：**

| idea 组件 | GeoMM 已做 | 剩余空间 |
|---|---|---|
| 测地距离替换余弦 | ✅ 是 | ❌ 几乎无首发性 |
| 聚类中心上跑最短路 | ✅ 分层 K-means | ⚠️ 细节差异，不足以立文章 |
| 全局结构感知 | ❌ 未做"轮廓对齐目标" | ✅ **有**（与 InfMasking 掩码目标结合是新的） |
| 与掩码/协同学习结合 | ❌（其框架是 ALBEF：ITC/MLM/ITM） | ✅ **有**（掩码协同 + 全局轮廓 = 新组合） |
| 协同任务（synergy）专属评测 | ❌ 无（检索/VQA 为主） | ✅ **有**（Trifeature synergy 评测是 InfMasking 系特有） |

**结论**：不能以"用测地距离替换余弦"为第一卖点（审稿人必引 GeoMM 拒稿）。**建议把卖点从"度量替换"转向"用全局/原型级关系结构训练协同学习（synergy）"**——这是 GeoMM 与 InfMasking 都未覆盖、且与投稿目标（Multimodal Fusion / 协同信息）一致的定位：

> **卖点候选**：协同信息是**跨模态共享的全局关系结构**，局部对比（batch 内、余弦）无法捕捉——UniGIR 用"原型级全局关系轮廓对齐 + 掩码协同学习"显式建模之；测地距离只是"度量更匹配流形"的增强选项（与 GeoMM 对照而非对立）。
> 若成立，Trifeature synergy (77.0) 与真实数据 synergy 任务（hateful memes / sarcasm / MUSTARD）是最有区分度的评测场，而非检索类任务（那是 GeoMM 的主场）。

---

## 二、资产盘点

### 已有代码（本地已跑通）
- **InfMasking 仓库**（工作区根 = 复现源，Windows 已适配）：`../main_trifeatures.py`、`../main_multibench.py`、`../main_multibench_reg.py` 三个入口；模型 CLIP/CrossSelf/CoMM/**InfMasking** 齐备；Trifeatures 数据集已生成。
  - **注意**：本地跑通的只是 Trifeatures 的"红绿灯验证"；MultiBench 真实数据（MOSI/UR-FUNNY/MIMIC/MUSTARD 等）**尚未下载**，`../dataset/data/` 内各真实数据集子目录为空。MMIMDb/Hateful Memes 入口有代码但无数据。
- **Indra 仓库**（`../Indra`，NeurIPS 2025 官方代码）：**MS-COCO/NOCAPS/OfficeHome 类（VLM/ALM）**评测体系，与 InfMasking 的数据集体系不重叠。

### 计算资源
- **本机：无 GPU**。CPU 实测：embed512 训练 step ≈36 s → 单 epoch ≈1.6 h。官方完整配置（100 epochs×5 seeds×2 组）纯 CPU 不可行。**完整实验必须在 GPU 上跑**，这是当前第一阻塞项。

### 时间线对照
idea 文档时间线以"2026-09 开始复现 + 实现"为前提，而现状是：InfMasking 单机复现刚通（1 epoch 验证），GPU 未落实。**若 9 月无法上 GPU，11-12 月 MultiBench 全量实验必延期**。建议将 ACM MM 2027 视为可达成目标，但需要尽快决策资源。

---

## 三、技术可行性审查（对 idea 方案中硬伤的前瞻）

| # | 问题 | 严重度 | 分析与建议 |
|---|---|---|---|
| 1 | **"从 $x_i$ 到聚类中心 $c_k$ 的测地距离 $d_g(z,c_k)$"未定义** | 🔴 概念硬伤 | 测地距离只定义在图上的**节点之间**。新样本/batch 内样本不在图上 → 必须**先 attach 到图**（GeoMM：挂到最近的下层聚类中心；或 k-NN 到中心后经图最短路）。idea 公式与 3.2/3.3 均直接写 $d_g(z_i,c_k)$，缺 attach 步骤定义 |
| 2 | **batch 内测地距离：$d_g(z_a,z_b)$ 同样未定义**（(a) 部分"局部对齐用测地 InfoNCE"） | 🔴 同上 | batch 内仅 B≈64-128 个点，无法建可靠 k-NN 图。可行替代：**测地 = 样本经最近中心 attach 后的图距**（GeoMM 方案），或放弃 batch 内测地、仅用余弦/欧氏做局部对齐，把测地用在"原型级轮廓"层面 |
| 3 | **训练中特征空间漂移 vs 离线原型** | 🟡 严重但可解 | 离线阶段用冻结预训练编码器建图；训练时该编码器不更新（idea 3.1 写明冻结 → 可）。但可训练编码器输出空间与原型空间**不在同一空间**，距离无意义 → 需原型/图挂在**不参与反向传播的 EMA 编码器输出**上（BYOL 式），或挂在融合层输出但 stop-gradient + 定期（每 T₀ 步）重聚类/重构图（GeoMM 的增量更新） |
| 4 | **"跨模态轮廓一致"（同事物不同模态 → 轮廓相似）的假设** | 🟡 需验证 | 图基于哪一层的表征建？若多模态各自建图，模态间轮廓可比性存疑；若共同表征（融合层）建图，单模态推理时无共同空间。需在 Phase 1 用 InfMasking 的 z1/z2/mask 输出实测轮廓一致性 |
| 5 | **K 与 batch 匹配**：$O(K^2)$ 矩阵存储/传输、K 大小与数据集量级 | 🟢 可工程解决 | K=500-2000 时 K²=0.25-4M 浮点/batch；批量矩阵运算可行；注意 DDP 下原型需全局同步（all-gather 后每 rank 各自 attach 或用共享图） |
| 6 | **效率对比缺失**：global 轮廓的"图成本" vs Indra O(n²) vs GeoMM 的分层图 | 🟢 | 论文需给出与 GeoMM 的复杂度/显存对照表 |
| 7 | 消融/评测口径要与 **InfMasking 77.0 synergy（Trifeature）** 严格对齐 | 🟡 | 我们已验证其 probing 流程可跑，但完整对照需其原始超参与 seed 范围 |

**对应设计修正方向（写进 Phase 1 就做）**：把"测地距离"重新定义为**原型级图上的最短路径距离，样本先经最近中心（或 k-NN 混合）attach 到图**——即把 GeoMM 的 attach 机制与 idea 的"全局轮廓"合并成一个自洽的数据流。三种变体可并行设计、用 synergy acc 消融择优：
- **V1**：attach 权重（样本到中心距离）+ 图距离 → 软轮廓
- **V2**：纯"到 K 个中心的**欧氏/余弦**距离"作轮廓（无图，最简基线，能单独验证"全局结构对齐"这一卖点本身）
- **V3**：GeoMM 式真·测地轮廓（含重构图），验证"测地 > 欧氏"是否有增量

---

## 四、总体路线

> 原则：**主基线（InfMasking）已复现 ✅ → 最短路径验证新损失可行 → 再上真实数据与完整实验**。全程默认 GPU 优先；若 9 月无 GPU，则用"减配版（embed 256/batch 32，单 step <10 s）"在 CPU 跑方法与消融，GPU 到位后补全量。

### Phase 0：资源与基线加固（~1 周，可并行）
- [x] **卖点定位调整**：✅ 已确认——从"测地首发性"转向"全局关系结构感知的协同学习"（2026-09-09 决策）
- [ ] **GPU 决策**：✅ 已确认暂缓 GPU，**CPU 减配先行**（embed 256/batch 32，单 step <10 s）；GPU（AutoDL/Colab/实验室）待有需要或预算时再落实，规划不变，只是执行节奏拉长
- [ ] 与导师汇报现状（见 `导师汇报与求助事项.md`），同步**创新定位调整**
- [ ] 备份当前已跑通的 Windows 适配改动；梳理 MultiBench 各数据集的下载来源与许可
- [ ] 对齐 InfMasking 论文数字：Trifeature 上官方超参（100 epochs 等）跑通≥1 seed，记录其 share/unique/synergy probing 数值作为对照基线

### Phase 1：方法验证（Trifeature 显微镜，预计 3-6 周 GPU）
- [ ] 实现 **全局轮廓对齐损失 L_global（V2 最简版：原型 + 欧氏/余弦轮廓）** 挂在 InfMasking 上：离线冻结预训练编码器聚类建原型；训练时 EMA/stop-grad 输出经 attach 得轮廓；掩码视图轮廓对齐完整视图轮廓（可选：加入"预测原型分配"的对比项）
- [ ] 实现 **测地增强版（V3）**：原型 k-NN 图 + Floyd + 定期重构图；对比 V2/V3 与纯 InfMasking 的 synergy acc
- [ ] 消融矩阵（按 idea 6.4）：L_global 有无、测地 vs 欧氏/余弦、K 敏感性（50→5000）、掩码比例×度量交互
- [ ] 概念验证关卡：**在 synergy 上显著 > 基线（77.0）且消融方向合理**；否则回头调 V1/V2/V3 或弱化"测地"成分、强化"全局结构"成分
- [ ] 鲁棒性快检：丢模态 / 加噪的 acc 下降曲线（低代价、高叙事价值）

### Phase 2：真实数据（MultiBench + 协同类数据集，预计 6-10 周 GPU）
- [ ] **synergy 区分度最高的数据集优先**：hateful memes（讽刺=协同）、MUSTARD/UR-FUNNY（幽默/讽刺）、MIMIC（临床多模态）——这些比"检索"更能证明协同卖点；检索类（MS-COCO）作为可选项与 GeoMM 正面对照
- [ ] MultiBench 全量入口跑通 + 与官方表（InfMasking 已报 67.05 平均等）对齐
- [ ] 与 GeoMM 的对照：在共享数据集上同口径复现对比（GeoMM 代码公开）；如资源有限则至少在 MS-COCO 检索一项上对照
- [ ] 每数据集 ≥3 seeds、记录均值±std

### Phase 3：写作与投稿（2027-01 起，与 Phase 2 尾重叠）
- [ ] 热力图（原型测地矩阵）、轮廓 t-SNE、鲁棒性曲线、复杂度对照表
- [ ] 初稿 → 导师审 → 润色；ACM MM 2027 截稿（预计 2027-03/04）前 2 周定稿

### 里程碑总览

| 时间 | 里程碑 | 关卡 |
|---|---|---|
| 2026-09 | Phase 0 | GPU 落实；定位与导师对齐 |
| 2026-10 | Phase 1 过半 | Trifeature synergy > 77.0（任意变体） |
| 2026-11 | Phase 1 完成 + Phase 2 启动 | 消融完整、V1-V3 定版 |
| 2026-12 | Phase 2 过半 | MultiBench 首批结果 + 对照基线 |
| 2027-01 | Phase 2 完成 | 全量实验与可视化齐 |
| 2027-02/03 | Phase 3 | 初稿/修改 |
| 2027-03/04 | 投稿 ACM MM 2027 | — |

---

## 五、风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| **GeoMM 高度重叠导致 novelty 被判不足** | 高 | 卖点转向"协同 + 全局结构 + 掩码"组合与评测（synergy 任务），论文中正面 cite 并差异化 GeoMM；Phase 1 就验证 synergy 增益 |
| **CPU-only 拖期** | 高（已发生） | 立即落实 GPU（AutoDL 按量成本低）；CPU 只跑减配版方法与消融 |
| 全局轮廓在真实数据上无增益 | 中 | Phase 1 显微镜先证伪/证实；备选：把"轮廓预测"换成"原型判别/聚类一致性"等其他全局目标，框架不变 |
| 原型-特征空间漂移导致训练不稳定 | 中 | EMA 编码器 + 定期重聚类（GeoMM 同款增量策略）；Phase 1 早测 |
| MultiBench 真实数据下载/许可受阻 | 中 | 优先公开/易得数据集；MMIMDb 需注册（导师或已有数据） |
| ACM MM 2027 截稿前实验未齐 | 中 | 保底：投稿范围缩至 Trifeature + 2-3 个真实集；再不行转投 2027 下半年会议（时间线余量保留到 2027-04） |

---

## 六、下一步行动（本周）

1. ~~用户决策~~：✅ GPU=暂缓/CPU 减配先行；✅ 卖点=接受转向（"全局关系结构感知的协同学习"，测地作对照增强项）
2. **导师沟通**：用《导师汇报与求助事项.md》同步进度 + 定位调整；GPU 需求保留为"后续如提速可申请"
3. **Phase 1 前置设计**（CPU 减配版即可开始）：
   - 起草 UniGIR **方法定稿文档**：V1/V2/V3 的精确损失公式、原型 attach 方案、EMA/stop-grad 机制
   - 跑通 InfMasking **减配基准**（embed 256/batch 32）并记录各任务 probing 基线数字
   - 验证 idea 关键假设（第 4 号硬伤）：**跨模态轮廓一致性**是否成立——用现成 InfMasking 输出的多模态表征实测"同一语义不同模态 → 轮廓是否近似"

## 七、已记录决策（2026-09-09）

| 决策点 | 结论 |
|---|---|
| 卖点定位 | **转向**：全局关系结构感知的协同学习（测地距离降级为与 GeoMM 对照的增强项） |
| 计算资源 | **暂缓 GPU**：先以 CPU 减配（embed 256 / batch 32）跑方法与消融；GPU 按需再租 |
| 投稿目标 | ACM Multimedia 2027（Multimodal Fusion），时间线保留弹性 |

---

## 八、Phase 1 实现进展（2026-09-09，V2 已落地）

### 代码改动（工作区，未提交）
| 文件 | 内容 |
|---|---|
| `../losses/prototype_alignment.py`（新增） | PrototypeAlignment：K 个 L2 归一化原型 buffer；Sinkhorn-Knopp 均衡软分配（3 轮行列归一）；掩码视图 → 完整视图码的 KL 对齐（`_predict`）；EMA 原型更新（只在训练态，避免验证污染）；梯度安全（前向用 detach 快照、EMA 后置） |
| `../losses/infmasking_loss.py`（扩展） | `InfMaskingLoss.__init__` 增 `profile_kwargs=None`；forward 尾部可选分支：`z1[prototype]`（完整多模态视图）与 z2 的完整视图做 prototype codes，6 个掩码视图做 KL 对齐；loss 加权并入总 loss；log `ssl_acc_profile` |
| `../configs/model/unigir.yaml`（新增） | `+model=unigir`：即 infmasking 配置 + `loss_kwargs.profile_kwargs`（dim=256 对齐 head 输出、K=512、τ=0.1、EMA 0.05、cross=false）。不改模型结构，`profile=false` 时与原 InfMasking 完全一致 |

### V2 设计要点（与 idea 的对应）
- idea 需"冻结预训练编码器 + 离线 k-means 建原型"→ 从头训练场景不可用，改为 **在线 SwAV 式原型库**（EMA + Sinkhorn 均衡分配），等价效果：每样本的"全局关系轮廓"= 对 K 个原型的软分配
- idea 的"掩码视图应能判断自己在全局拓扑中的位置"→ 损失：`KL( codes(full) || softmax(mask@P/τ) )`，六个掩码视图同时监督
- 测地距离（V3）暂未实现——先验证"全局轮廓对齐"本身是否有增益（消融 1 的对照组）

### 冒烟结果（CPU，1 epoch，max_size=1024）
- 训练正常收敛无 NaN；`val_ssl_acc_profile=0.184`（K=512 随机水平 0.2%，已捕获结构）
- 下游 probing 正常：share 0.353 / unique1 0.244 / unique2 0.158 / synergy 0.5（1-epoch 模型合理水平）
- 4 个下游任务单轮 probing 在 CPU 约 15 分钟（max_size=1024 时）

### 进行中
- [ ] **对照演示**（后台运行中）：baseline（infmasking）vs UniGIR（unigir），seed 42 / 4 epochs / biased=false / max_size=1024 → `../_demo_baseline.log`、`../_demo_unigir.log`
- [ ] 对照完成后：acc1 曲线对比 → 决定 V3（测地轮廓）与 K/τ/λ 消融方向

### 对照演示结果 1/2：Baseline（InfMasking，4 epochs，biased=false，max_size=1024，seed 42）

Linear probing acc@1 随 epoch（下游 Sup 评测）：

| epoch | share | unique1 | unique2 | synergy | mean acc1 |
|---|---|---|---|---|---|
| 0 | 0.335 | 0.206 | 0.175 | 0.500 | 0.304 |
| 1 | 0.368 | 0.279 | 0.209 | ~0.50 | 0.338 |
| 2 | 0.422 | 0.274 | 0.215 | 0.497 | 0.353 |
| 3 | 0.415 | 0.281 | 0.288 | 0.504 | **0.372** |

- 收敛正常：mean acc1 0.304→0.372；synergy 在 biased=false 下保持 ~0.5（随机，符合 R-U 实验设计）
- 日志：`../_demo_baseline.log`；产物：`../InfMasking/bimodal_trifeatures/R-U-cpu-baseline/logs/version_0/`
- 运行事故教训：双 python 进程互抢 16 核导致假死 → **启动前确认无残留 python、单进程串行跑**
- 待跑：UniGIR（unigir 配置）对照组 → `../_demo_unigir.log`（运行中，~2h）

### 对照演示结果 2/2：UniGIR vs Baseline（4 epochs，同设置）

| epoch | Baseline share/u1/u2 | mean acc1 | UniGIR share/u1/u2 | mean acc1 |
|---|---|---|---|---|
| 0 | 0.335/0.206/0.175 | 0.304 | 0.322/0.219/0.194 | 0.309 |
| 1 | 0.368/0.279/0.209 | 0.338 | 0.401/0.239/0.202 | 0.339 |
| 2 | 0.422/0.274/0.215 | 0.353 | 0.422/0.245/0.213 | 0.345 |
| 3 | 0.415/0.281/0.288 | **0.372** | 0.411/0.277/0.234 | **0.355** |

synergy 两组均在 ~0.5（biased=false 下随机，符合设计）。

**Profile 分支自身的学习信号（机制有效）**：`val_ssl_acc_profile` = 0.159 → 0.316 → 0.317 → **0.373**（K=512 时随机仅 0.2%）——掩码视图预测完整视图全局原型位置的能力持续上升。

**初步结论（诚实版）**：
- ✅ 机制工作：原型库学习正常、profile 预测准确率上升、训练稳定无 NaN
- ⚠️ 短跑（4 epoch/单 seed/max_size=1024）**下游无增益甚至略逊**（0.355 vs 0.372，差异在评测噪声内）
- 可能原因：①4 epoch 太短，原型库 EMA 收敛慢；②synergy 区分度需 **biased=true（S 组）** 才能体现——R-U 组 synergy 恒 0.5；③profile 损失权重均分 1/9 过小；④下游 max_size=1024 评测噪声大
- **下一步方向**：S 组（biased=true）对照（真正测试协同信息）> 提高 profile loss 权重/更长训练 > K/τ 消融

### 对照演示 3/3：S 组（biased=true，synergy 场景）— 4 epochs，seed 42

| epoch | S-Baseline share/u1/u2/syn | mean | S-UniGIR share/u1/u2/syn | mean |
|---|---|---|---|---|
| 0 | 0.359/0.211/0.216/**0.507** | 0.322 | 0.337/0.210/0.226/**0.500** | 0.318 |
| 1 | 0.340/0.236/0.204/**0.500** | ~0.320 | 0.382/0.215/0.242/**0.504** | 0.336 |
| 2 | 0.363/0.270/0.232/**0.507** | 0.343 | 0.376/0.249/0.214/**0.500** | ~0.335 |
| 3 | 0.398/0.343/0.243/**0.517**(roc 0.566) | **0.375** | 0.366/0.267/0.246/**0.500** | **0.345** |

- S 组 synergy 终于露头：baseline ep3 达 0.517/roc 0.566（首个 >0.5 信号）→ **synergy 确实需要长训练**（论文 100 epochs 达 77.0），4 epochs 不足以判优劣
- UniGIR profile 机制在 S 组同样学习（val_ssl_acc_profile 0.241→0.350）
- 日志：`../_demo_S_baseline.log`、`../_demo_S_unigir.log`

### 四组总览（4 epochs CPU，seed 42，max_size=1024）——诚实结论
| 组 | Baseline mean acc1 | UniGIR mean acc1 | 差异 |
|---|---|---|---|
| R-U (biased=false) | **0.372** | 0.355 | -0.017 |
| S (biased=true) | **0.375** | 0.345 | -0.030 |

**判断**：
1. profile 机制在两组都有效学习（预测准确率远超随机且上升），训练稳定
2. 但 4-epoch 短跑两组均**未见下游增益、方向性略逊**（差 0.02-0.03，处于评测噪声量级但两组同向，需警惕）
3. synergy 任务 4 epochs 学不出来 → **V2 增益的判定需要 30+ epochs（GPU）**；CPU 上单组 100 epochs ≈ 40h+，不可行
4. 现有证据不足以支持 V2 增益假设 → 决策点：GPU 长训验证 / 调整设计（α、K、原型空间）/ 或重新审视"轮廓对齐"目标形式
