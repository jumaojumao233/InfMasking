# UniGIR

# UniGIR: Unifying Global Relational Profiles and Infinite Masking for Synergistic Multimodal Learning

> v2 方案（2026-09-09 修订）。前身见同目录 `idea.md`。本版吸收两轮调研结论：① 测地距离替换余弦已被 GeoMM (CVPR 2025) 抢先；② R-U 组实验无法体现协同增益，需以 S 组为主战场。修订说明见 `UniGIR_idea修订与导师汇报.md`。

## ACM Multimedia 2027

| 论文                                                         | 链接                   | 关系                                                         |
| ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ |
| InfMasking: Unleashing Synergistic Information by Contrastive Multimodal Interactions (NeurIPS 2025 Spotlight) | proceedings.neurips.cc | 本工作的底座（掩码协同框架），已复现                         |
| The Indra Representation Hypothesis for Multimodal Alignment (NeurIPS 2025) | proceedings.neurips.cc | 关系轮廓思想的来源（样本级、推理用）                         |
| GeoMM: On Geodesic Perspective for Multi-modal Learning (CVPR 2025) | openaccess.thecvf.com  | 测地距离已由其发表，本工作降级为可选增强，改为主打全局结构   |
| SwAV (NeurIPS 2020)                                          | proceedings.neurips.cc | 在线原型加 Sinkhorn 分配的技术先例，本工作在掩码协同场景复用 |

---

### 一、基本信息

**题目（暂定）**：UniGIR: Unifying Global Relational Profiles and Infinite Masking for Synergistic Multimodal Learning

**投稿会议**：ACM Multimedia 2027（Thematic Area: Multimodal Fusion）

**核心贡献**：我提出 UniGIR，指出多模态协同学习中的一个结构性缺失：协同信息本质上是跨模态共享的全局关系结构，而现有方法只在小批量内用局部度量（余弦）建模样本关系，既没有全局参照系，也无法在输入残缺时保持对自身全局位置的判断。UniGIR 用原型级全局关系轮廓对齐（掩码视图预测完整视图在 $K$ 个原型上的软分配）把 Indra 的全局关系思想从 $O(n^2)$ 推理后处理变成 $O(B\cdot K)$ 的端到端训练目标，并保留 InfMasking 的无限掩码协同框架；在此基础上提供测地距离作为可选增强（原型图最短路径，与 GeoMM 对照而非对立）。我在 Trifeature 与 MultiBench 等基准上验证协同信息捕捉能力。

### 二、Introduction：协同学习缺的不是更准的局部度量，而是全局参照结构

多模态自监督学习（如 InfMasking）通过随机掩码和跨视图对齐，在捕捉跨模态协同信息上取得了显著进展。但审视现有方法，会看到一个共性的结构缺陷：

**第一，所有对比和对齐都发生在 batch 内。** 样本间的正负关系只由同一个小批量（通常 64 到 128 个样本）决定。batch 是采样的产物，不是数据的结构，模型从未见过这个样本在整个数据空间中处于什么位置。

**第二，协同信息是关系型信息，却缺少全局参照系。** 协同的定义是只有多模态结合才涌现的信息，它体现在样本之间、模态之间的相对关系上。要把这种关系变成可学习的信号，需要一组稳定的参照点（原型或地标），现有方法没有这种结构，模型对残缺输入应落在哪个语义区域没有约束。

**第三，几何上的次优是表层的，结构上的缺失是根本的。** GeoMM (CVPR 2025) 指出余弦相似度的球面假设与流形数据失配，并提出用测地距离替换（k-NN 图加聚类中心最短路）。这是一个有价值的修正，但它仍只作用于局部对之间用什么度量。换了一个更准确的局部度量，却没有建立全局参照系。我的主张是：协同学习的关键瓶颈是后者。

**我的工作。** 我提出 UniGIR：

1. 在掩码协同学习（InfMasking）之上引入 $K$ 个在线原型作为全局参照系，用 Sinkhorn 均衡分配为每个样本生成全局关系轮廓（对 $K$ 个原型的软分配，$O(B\cdot K)$）；
2. 新增全局轮廓对齐损失：每个掩码视图（残缺输入）必须预测其完整视图的轮廓，模型学会无论眼前多残缺，都知道自己在全局拓扑中的位置；
3. 保留 InfMasking 的局部对齐和跨模态损失，并提供测地轮廓（V3：原型图最短路径）作为度量增强项，用消融回答全局结构与更准度量各自的贡献。

### 三、方法的完整数据流

> 已实现部分标注 ✅（代码：`../losses/prototype_alignment.py`、`../losses/infmasking_loss.py`、`../configs/model/unigir.yaml`）。

#### 3.1 原型库的准备与维护

**目标**：得到 $K$ 个 L2 归一化的原型 $P=\{p_1,\dots,p_K\}\subset\mathbb{R}^d$（$d$ 为投影头输出维度，本文取 256），作为全局参照系。✅

**在线初始化与维护（本文默认，从头训练场景）**：原型与模型共同演化。

- 初始化：随机高斯，再 L2 归一化；
- 每个训练步：对 batch 内完整视图表征 $Z_{\text{full}}$ 计算 Sinkhorn 均衡分配矩阵 $Q\in\mathbb{R}^{B\times K}$（3 轮行列归一），再以 EMA 更新：

$$
P \leftarrow \mathrm{L2norm}\big(P+\eta\cdot Q^{\top}Z_{\text{full}}\big),\quad \eta=0.05
$$

**离线变体（可选，当预训练编码器可用时）**：冻结编码器提全量特征，k-means 得到原型后固定，与 v1 一致。

**测地增强（V3，可选）**：以 $K$ 个原型为节点建 k-NN 图，边权为欧氏距离；用 Floyd 或优先队列求全源最短路，得到 $K\times K$ 测地距离矩阵 $D_g$。样本 $z$ 经最近中心 attach 后，其到各原型的测地距离为

$$
\min_j\big(\|z-p_j\|+D_g[j][k]\big).
$$

定期（每 $T_0$ 步）重构图。⚠️ 未实现。

#### 3.2 训练阶段（核心）

**可训练模块**：各模态编码器 $f_{\text{mod}}$、融合模块 $f_{\text{fuse}}$、投影头 $h$，与 InfMasking 相同结构。

**前向**：对 batch 中每个多模态样本，生成两种视图：

1. **完整视图**：所有模态完整输入，

$$
z_i^{\text{full}}=h\big(f_{\text{fuse}}(f_{\text{mod}_1}(x_i^{(1)}),\dots,f_{\text{mod}_M}(x_i^{(M)}))\big)
$$

2. **掩码视图**（每个样本 $T$ 个，$T=6$）：随机遮挡各模态 70% 的 token，得到 $z_i^{\text{mask},t}$。✅

**损失函数分三个部分。**

**(a) 局部对齐和跨模态损失**（继承 InfMasking，✅ 未改动）

- 掩码视图与完整视图的 InfoNCE 类对齐（含模态内和模态间），记为 $\mathcal{L}_{\text{local}}$；
- 沿用其分布匹配加对比的 mask loss 结构，温度 $\tau=0.1$，$\lambda_{\text{mask}}=1$。

**(b) 全局轮廓对齐损失（核心创新）✅**

对每个样本 $i$：

1. 完整视图的**全局关系轮廓**等于 Sinkhorn 均衡软分配（伪标签，不反传）：

$$
q_i=\mathrm{Sinkhorn}\big(Z_{\text{full}}P^{\top}/\varepsilon\big),\quad q_i\in\Delta^K
$$

2. 每个掩码视图预测该轮廓：

$$
p_{i,t}=\mathrm{softmax}\big(z_i^{\text{mask},t}P^{\top}/\tau_p\big)
$$

3. KL 对齐，$T$ 个视图取平均，两组增广各算一次：

$$
\mathcal{L}_{\text{global}}=\frac{1}{2}\sum_{a\in\{1,2\}}\frac{1}{T}\sum_{t=1}^{T}\mathrm{KL}\big(q_i^{(a)}\,\|\,p_{i,t}^{(a)}\big)
$$

**作用**：残缺输入（掩码视图）必须还原自己在全局原型空间中的位置，把全局拓扑感知变成训练目标的一部分。

**(c) 可选测地增强（V3）⚠️**：若启用，将轮廓的相似度矩阵由余弦改为经原型图计算的测地相似度

$$
s_{\text{geo}}(z,p_k)=-(d_g(z,p_k)+\epsilon)^{-1},
$$

与 V2 软分配轮廓做消融。

**总损失**：

$$
\mathcal{L}_{\text{Total}}=\mathcal{L}_{\text{InfMasking}}+\alpha\mathcal{L}_{\text{global}},\quad \alpha \text{ 可调，默认 }1.0
$$

**反向传播**：三部分统一反传。原型本身不接收梯度（buffer 加 EMA 更新），轮廓伪标签 detach，梯度只流向掩码视图表征及其后的编码器与融合模块。✅

#### 3.3 推理阶段

- **特征输出** $z_{\text{new}}$：冻结 backbone，接轻量线性头用于分类；✅（评测链路等于 InfMasking 的 linear probing）
- **关系轮廓输出**：$\mathrm{Profile}(z_{\text{new}})=\mathrm{softmax}(z_{\text{new}}P^{\top}/\tau_p)\in\mathbb{R}^K$（推理时 $P$ 已固定），可用于检索（轮廓相似度排序）和跨模态对齐（不同模态样本若轮廓相近则语义对齐）。✅（V2 形态）

流程图：

```
【训练阶段】—— 对每个 batch 循环

Batch 数据（B 个多模态样本）
        ↓
各模态编码器 + 融合模块 + 投影头（可训练）
        ↓
完整视图 z_full (B×d)        掩码视图 ×T (T·B×d)
        ↓                          ↓
Sinkhorn 均衡分配 → 轮廓 q (B×K)   预测 p_t = softmax(z_mask Pᵀ/τ)
        │        （detach 伪标签）       ↓
        └──── KL(q ‖ p_t) ←────────────┘   (L_global)
                 ↓
        原型 EMA 更新 P ← norm(P + η·Qᵀ z_full)（仅训练态）
                 ↓
总损失 = L_InfMasking(局部对齐+跨模态) + α·L_global → 反传编码器/融合器
```

```
【推理阶段】
新样本 x → 冻结编码器+融合 → z ∈ R^d
        ├──→ 分类：z 或 Profile(z) → 轻量线性头
        └──→ 检索/跨模态对齐：Profile(z) 与库中轮廓做相似度排序
```

### 四、方法的直观理解

- **InfMasking** 像一个只跟同桌对答案的学生：给一张看不清的卷子（掩码视图），让他猜同桌（batch 内完整视图）写了什么。他永远不知道自己在全年级的位置。
- **GeoMM** 把这个学生手里的直尺换成山路距离地图，量得更准，但他依然只对同桌说话，依然没有全校坐标系。
- **UniGIR** 先立起 $K$ 个原型作为参照点，让每个学生都学会报出自己离 $K$ 个原型各有多远（关系轮廓）。考试时即使题目残缺（掩码），也要能报出同样的坐标。残缺输入仍应保持全局位置，这本身就是协同信息（跨模态信息补全加全局定位）。

### 五、与已有工作的关系

| 工作                          | 核心方法                                        | 本工作的关系                                                 |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| **InfMasking (NeurIPS 2025)** | 无限掩码加余弦对比，捕捉协同                    | 底座：掩码协同框架完整保留；新增全局轮廓目标                 |
| **Indra (NeurIPS 2025)**      | Yoneda 嵌入，样本级关系轮廓（$O(n^2)$，推理用） | 将其思想原型级化（$O(B\cdot K)$）并训练内化：由事后描述关系变为残缺输入必须预测关系 |
| **GeoMM (CVPR 2025)**         | 测地距离替换余弦（图加聚类中心最短路）          | 承认其测地先发性；本工作主张全局结构为主、测地为可选增强，并与之做消融对照 |
| **SwAV/DINO 系**              | 在线原型加 Sinkhorn/teacher 分配                | 技术先例；但应用场景不同：其为单模态增广一致性，本工作为掩码协同加跨模态，且轮廓是显式输出而非仅内部目标 |

**一句话区别**：SwAV 让同一张图的两个裁剪对齐到同一原型；UniGIR 让残缺的多模态输入对齐到完整多模态输入的全局轮廓。监督信号从视图不变性升级为协同补全加全局定位。

### 六、实验设计

#### 6.1 研究问题

- **RQ1**：全局轮廓对齐损失 $\mathcal{L}_{\text{global}}$ 能否提升协同信息的捕捉？（主战场：S 组 synergy）
- **RQ2**：加入 $\mathcal{L}_{\text{global}}$ 是否损伤冗余和独有信息？（R-U 组须不降）
- **RQ3**：轮廓对齐与掩码协同如何交互？（掩码比例乘 $\mathcal{L}_{\text{global}}$ 消融）
- **RQ4**：V2（余弦软分配轮廓）与 V3（测地轮廓）相比，全局结构和更准度量各自贡献多少？
- **RQ5**：$K$ 与 $\alpha$ 的敏感性、鲁棒性（丢模态或加噪）如何？

#### 6.2 Trifeature（小规模机制验证）

**设置**（沿用 InfMasking 官方配置，代码已就绪）：

- **S 组（biased=true，主实验）**：texture–color 人为相关，synergy 任务有信号，检验 $\mathcal{L}_{\text{global}}$ 增益；对照组为 InfMasking 官方 synergy $77.0\pm4.22$
- **R-U 组（biased=false，副实验）**：share/unique1/unique2 须不劣于基线，防止只对协同有效却丢冗余
- 100 epochs，至少 3 seeds（GPU）；CPU 环境先以 10 到 20 epoch 观察趋势

**已验证**：R-U 组 baseline 收敛正常（acc1 0.30 到 0.37 @4ep，seed 42）；V2 训练稳定，轮廓预测准确率随训练上升（0.159 到 0.373，随机 0.2%）。S 组对比为当前第一待办实验（本机 CPU 约 2h/组可先跑）。

#### 6.3 MultiBench（真实任务验证）

- 协同类优先：MUSTARD/UR-FUNNY（讽刺和幽默）、hateful memes（讽刺）、MIMIC（临床多模态）；检索类（MS-COCO）作为与 GeoMM 正面比较的可选项
- 基线：InfMasking（必须）、GeoMM（如资源允许）、CoMM、Indra（如可行）
- 指标：Acc、回归 EE、检索 Recall@K；至少 3 seeds

#### 6.4 消融

1. $\mathcal{L}_{\text{global}}$ 有无（S 组为主判据）
2. V2 与 V3（轮廓用软分配还是测地距离），回答 RQ4
3. $K$ 敏感性（64 到 4096）、$\alpha$（0.1 到 5）
4. 掩码比例与 $\mathcal{L}_{\text{global}}$ 交互（0.5/0.7/0.9）
5. 原型更新率 $\eta$、Sinkhorn 轮数

#### 6.5 可视化与鲁棒性

- 原型分配混淆热力图、轮廓 t-SNE（同类样本轮廓应聚类）
- 丢模态和高斯噪声（$\sigma=0.5/1.0/2.0$）的 acc 下降曲线，预期 $\mathcal{L}_{\text{global}}$ 使下降更平缓（残缺输入已见过并学会报位置）
- 复杂度对照表（与 Indra 的 $O(n^2)$、GeoMM 分层图对比）

### 七、创新点

1. **问题层面**：指出协同学习的结构性瓶颈不是局部度量不准，而是缺乏全局参照系。batch 内对比无法表达样本在全局拓扑中的位置这一协同信息的载体。
2. **方法层面**：将 Indra 的关系轮廓原型级化并训练内化（$O(n^2)$ 推理后处理变为 $O(B\cdot K)$ 训练目标）：掩码视图必须预测完整视图的全局轮廓，使残缺输入到全局定位成为显式监督。
3. **度量层面**：不宣称测地首发性，而是把测地（GeoMM 式）作为可选增强并与之消融，回答全局结构与更准度量的归因问题，这本身是现有工作未做的对照。
4. **评测层面**：以 synergy 任务为主战场（S 组），R-U 组检验不损伤性，辅以消融、鲁棒性、可视化证据链。

### 八、时间线（ACM MM 2027；✅=已完成）

- **2026-09（现在）**：✅ 复现 InfMasking（Trifeature 全流程）；✅ V2 实现与首轮 R-U 对照；◻️ 跑 S 组对照（优先，CPU 2h/组可先行）→ 与导师确认定位（见第九节）
- **2026-10**：◻️ S 组和 R-U 组多 seed 完整验证；V3 测地实现；$K$/$\alpha$/掩码比例消融（Trifeature）
- **2026-11~12**：◻️ MultiBench 全量（MOSI/UR-FUNNY/MUSTARD/MIMIC 等）加 GeoMM 对照（检索可选）；鲁棒性和可视化
- **2027-01**：◻️ 初稿（含复杂度分析、效率对照）
- **2027-02~03**：◻️ 导师修改、补实验
- **2027-04 初**：◻️ ACM MM 2027 投稿

### 九、拿不准的点

下面三个问题想请导师帮我判断，每条选择一个选项。

1. **核心卖点怎么摆？**
   A. 以全局轮廓对齐为主，测地距离只作可选增强，并和 GeoMM 做归因对照。
   B. 以测地轮廓为主，全局轮廓作为辅助。
   C. 两者并重，分别写成两个贡献。
   我倾向 A，因为 GeoMM 已发表测地距离，直接对抗风险高；但担心审稿人认为 A 的新意不够强。

2. **第一优先级实验？**
   A. 先在 Trifeature S 组验证 $\mathcal{L}_{\text{global}}$ 对 synergy 的增益。
   B. 先上 MultiBench 的 MUSTARD/UR-FUNNY，直接看真实任务。
   C. 先补 V3 测地实现，再做完整消融。
   我倾向 A，因为成本低、机制清楚；但担心只有 Trifeature 不够有说服力。

3. **投稿节奏？**
   A. 按 ACM MM 2027 时间线推进，2027-04 投稿。
   B. 先投一个更早的会议或 workshop，再扩展期刊。
   C. 推迟到 2027 年下半年，补更多基线。
   我倾向 A，但担心 MultiBench 与 GeoMM 对照实验量过大。

### 十、总结

UniGIR 的核心主张是：多模态协同学习的下一个瓶颈，不是尺子不够准，而是没有全局参照系。现有方法在小批量内用局部度量对比样本，既无法感知全局拓扑，也无法在输入残缺时保持对自身位置的判断。我以 $K$ 个在线原型为参照系，把掩码视图必须预测完整视图的全局轮廓作为显式训练目标，在保留 InfMasking 掩码协同框架的同时，将 Indra 的全局关系思想训练内化，并以测地距离作为可选的度量增强与 GeoMM 归因对照。

需要说明的是：本文所有预期增益均以 Trifeature S 组 synergy 实验为首要验证关卡；当前已确认机制可学习（轮廓预测准确率随训练显著上升）、训练稳定、R-U 组不损伤（首轮持平），但协同增益尚未证实。S 组实验结果是本方案可行性的第一道闸门。