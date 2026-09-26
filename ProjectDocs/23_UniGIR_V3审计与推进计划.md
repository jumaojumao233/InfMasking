# UniGIR V2 与 V3 审计及推进计划

更新时间：2026-09-26

## 0.1 2026-09-26：修复版 V3 已完成，当前结果低于 V2

V3 初轮的 geodesic Sinkhorn assignment 因指数化下溢而无效。修复后，独立运行 `winpc_g5_v3_geodesic_s42_fix1` 已于 2026-09-26 10:32:47 以 `EXIT_CODE=0` 完成，`last.ckpt` 存在，GPU 已空闲。profile 分支的 train/validation `loss_profile` 为 `0.1007/0.1446`，`profile_kl` 为 `0.3638/0.5309`，active prototypes 为 `51.0/45.68`；graph 连通分量为 `1`，不可达 pair 为 `0`。

在 seed=42、10 epoch、同一 G0 数据、queue、$\alpha$ 和 probing 协议下，V3-Geodesic 修复版的四任务平均 Acc/AUC 为 `0.7385/0.9289`，V2-Cosine 为 `0.7615/0.9368`。V3 相比 V2 的平均差值为 `-0.0229/-0.0079`；synergy AUC 差值为 `-0.0229`，unique2 Acc 差值为 `-0.0316`。

这组结果证明修复后的 V3 可以正常训练，但没有在当前协议下带来信息增益。由于只有一个有效 seed 且预算为 10 epoch，当前不能把它写成跨 seed 的最终否定；同时也没有依据继续扩大 graph、temperature、EMA 或 loss weight 搜索。默认路线是保留 V2 主线，把 V3 作为有效的负向对照；如需增强统计证据，只增加预先指定的独立 seed，并固定其余设置。

## 0. 2026-09-26：V3 smoke 与短跑安排

V3 已完成 1 epoch smoke。smoke 使用 G0 封存数据、seed=42、`max_size=1024`、GPU 0 和独立运行目录，退出码为 0，`last.ckpt` 已生成。TensorBoard 记录的 prototype graph 指标为：连通分量数 `1`、不可达 pair 数 `0`、平均度约 `9.47`、平均边权约 `1.329`。这说明当前 `graph_k=8`、对称 k-NN、`graph_anchors=4` 的第一版实现可以完成训练和验证流程。

随后已启动 V2-Cosine 与 V3-Geodesic 的 10 epoch 成对短跑。两项固定使用 G0 数据、model seed=42、pair seed=42、`max_size=10000`、queue=1024、$\alpha=0.25$、`cross=false` 和 `by_fit` probing；V2 先运行，V3 由每 15 分钟运行的 G5 watchdog 自动接续。当前只有 smoke 工程结果，没有 V3 性能结论。

## 1. 审计结论

这份完整 idea 的三层结构是清楚的：InfMasking 提供掩码协同学习，在线 prototype 提供全局参照系，prototype graph 上的 shortest path 提供流形关系。这个结构可以作为论文方法路线，但当前实现只覆盖前两层。

当前代码已经实现：

- InfMasking 的完整视图、多个掩码视图、InfoNCE 和 mask loss；
- $K=128$ 个在线 prototype；
- 基于 cosine similarity 的 Sinkhorn 完整视图轮廓；
- 掩码视图到完整视图轮廓的 KL 对齐；
- assignment-weighted EMA prototype 更新；
- feature queue、profile loss weight 和 profile 诊断指标。

当前仍未完成：

- V3 的训练和下游验证；
- V2-Cosine 与 V3-Geodesic 的同协议成对实验；
- graph 更新频率、连通性和 anchor 数量的实验边界；
- V3 对 unique2 退化是否有改善的证据。

因此，当前研究位置应记为：

| 层次 | 当前状态 | 证据 |
|---|---|---|
| InfMasking | 已实现并完成基线验证 | 原始 InfMasking 训练、linear probing 和 checkpoint 流程已跑通 |
| UniGIR V2-Cosine | 已实现并完成主要实验 | G1、G2、G0、MOSI 和 unique2 诊断已完成 |
| V2 机制解释 | 尚未闭环 | G0 synergy AUC 上升，同时 unique2 Acc@1 下降；profile 聚合指标没有给出单一原因 |
| UniGIR V3-Geodesic | 代码已实现，尚未训练验证 | 已有 prototype graph、shortest path 和 geodesic profile 单元测试 |

## 2. 对公式的检查

### 2.1 V2 余弦轮廓

完整视图和 prototype 的 cosine score 为：

$$
s_{ik}=\frac{z_i^{\mathrm{full}}p_k^{\mathsf T}}
{\lVert z_i^{\mathrm{full}}\rVert_2\lVert p_k\rVert_2}.
$$

当前实现先归一化表征和 prototype，因此代码中的矩阵乘法等价于上式。完整视图的 target profile 经过 batch 和 queue 上的 Sinkhorn，并从计算图中 detach；掩码视图通过 softmax 预测这个 profile。这个部分与 V2 代码一致。

### 2.2 测地轮廓的距离符号

如果 $d_G$ 表示非负测地距离，概率分布应使用距离的负值：

$$
q_{ik}^{\mathrm{geo}}
=
\operatorname{softmax}_k\left(-\frac{d_G(z_i,p_k)}{\tau_g}\right).
$$

距离越小，logit 越大，才会得到更高权重。原有写法

$$
-\left(d_G+\epsilon\right)^{-1}
$$

会使远距离对应的 logit 更接近零，存在排序方向错误的风险。这个式子不能直接用于实现。

### 2.3 Sinkhorn 不能在 V3 中被无意删除

只使用 `softmax(-d_G / tau_g)` 会把 prototype 的访问频率、graph 度数和距离结构混在一起。为了和 V2 保持可比，我建议 V3 也对完整视图的 geodesic score 使用同一套 balanced assignment：

$$
q_i^{\mathrm{geo}}
=
\operatorname{Sinkhorn}
\left(-\frac{d_G(z_i,p_{1:K})}{\tau_g}\right).
$$

掩码视图仍然使用普通 softmax 预测 target profile。这样 V2 与 V3 的主要差别是距离结构，不能把 Sinkhorn 是否存在作为混杂变量。

## 3. V3 的实现约束

### 3.1 prototype graph 的构造

在每次 graph 更新时，使用同一份 prototype snapshot：

$$
G=(P,E),\qquad p_k\in\mathbb{R}^{d}.
$$

prototype 已做 L2 normalization，因此第一版可以使用：

$$
w_{jk}=\lVert p_j-p_k\rVert_2
$$

或等价的单调 cosine 距离 $1-p_jp_k^{\mathsf T}$。V3 第一版只能选一种距离并固定，不能同时改变距离和 graph 构造规则。

每个 prototype 连接 $k_g$ 个近邻，建议先取 $k_g=8$。只保留单向 k-NN 容易产生不连通图，第一版应使用对称边：只要 $j$ 是 $k$ 的近邻，或 $k$ 是 $j$ 的近邻，就加入无向边。

### 3.2 最短路径和连通性

prototype 数量只有 128，第一版可以在 CPU 或 GPU 上对加权邻接矩阵使用 Floyd–Warshall，避免引入 NetworkX 依赖。未连接的边使用 $+\infty$，对角线为 0。

实现必须记录：

- 图的连通分量数；
- 最大有限 shortest path；
- 不可达 prototype 对的数量；
- graph 平均度和边权均值。

如果图不连通，不能静默把不可达距离设为零。可以先增大 $k_g$，或者加入一个足够大的 fallback edge；两种处理必须记录在配置中。

### 3.3 样本到 graph 的近似距离

对样本 $z_i$，先找最近的 $r$ 个 prototype 作为 anchors，再计算：

$$
d_G(z_i,p_k)
=
\min_{j\in\mathcal{N}_r(z_i)}
\left[
\lVert z_i-p_j\rVert_2+D_G(j,k)
\right].
$$

第一版建议 $r=4$。如果只使用单个最近 prototype，距离会受到离群 anchor 的影响；如果使用全部 prototype，则失去近似的计算意义。

### 3.4 graph 的更新时间和梯度边界

V3 的 prototype graph 不应在每个样本的反向传播路径中更新。第一版采用以下边界：

1. prototype graph 从当前 prototype buffer 的 detach snapshot 构造；
2. graph 和 shortest path 在每个 epoch 开始时更新一次；
3. graph 计算不接收梯度；
4. 完整视图的 geodesic target detach；
5. 掩码视图预测 geodesic profile，只有 masked branch 接收 $\nabla\mathcal{L}_{\mathrm{geo}}$；
6. prototype EMA 仍按 V2 规则更新，V3 第一版不同时修改 EMA 机制。

如果把 graph 每个 step 更新，会增加噪声和计算成本，也会让 V2 与 V3 的差异难以归因。每个 epoch 更新一次适合第一轮判断。

## 4. 与 GeoMM 的区别

我核对了 CVPR 2025 Open Access 页面。GeoMM 的确使用样本图、shortest path 和层次化 graph 来构造多模态 geodesic distance，并将其用于多模态学习中的关系建模。[GeoMM 官方论文页面](https://openaccess.thecvf.com/content/CVPR2025/html/Mei_GeoMM_On_Geodesic_Perspective_for_Multi-modal_Learning_CVPR_2025_paper.html)

因此，UniGIR 不能把使用 geodesic distance 本身写成主要贡献。UniGIR 可以检验的差异是：

| 方法 | graph 或距离的作用 | 训练监督 |
|---|---|---|
| InfMasking | 通过掩码视图恢复完整视图中的协同信息 | InfoNCE 和 mask loss |
| UniGIR V2-Cosine | 用 prototype cosine profile 作为全局关系监督 | 完整视图 profile → 掩码视图 profile |
| UniGIR V3-Geodesic | 用 prototype graph 上的 geodesic profile 作为全局关系监督 | 完整视图 geodesic profile → 掩码视图 geodesic profile |
| GeoMM | 用样本图和层次 graph 改进多模态距离与负样本关系 | 依其论文中的多模态距离学习目标 |

论文能够主张的方向应是：把 prototype graph 的流形关系转成完整多模态视图到掩码多模态视图的关系蒸馏目标，并将它接入 InfMasking。最终是否成立，必须由 V2/V3 成对实验决定。

## 5. 当前实验位置

当前 V2 的正式证据如下：

- G0 synergy ROC AUC 三个 seed 全部上升，平均 `+0.026`；
- G0 unique2 Acc@1 三个 seed 全部下降，平均 `-0.053`；
- G0 四任务平均 Acc@1 下降 `-0.010`，四任务平均 AUC 上升 `+0.003`；
- MOSI 三个 seed 的 Acc 和 AUC 都上升；
- unique2 固定变换下仍然下降，profile usage entropy 和 active prototype 数没有显示明显 collapse；
- alpha=0.125 在 MOSI 上低于 alpha=0.25，已经停止继续搜索 loss weight。

所以当前要回答的第一问题不是 V3 能不能提升结果，而是 V2 的 profile gradient 是否与 unique2 所需表征方向发生明显冲突。这个问题用现有 checkpoint 做只读梯度诊断即可，不需要新训练。

## 6. 已启动的下一步

第一步是 V2 梯度冲突诊断。诊断分别计算：

$$
g_{\mathrm{base}}=\nabla_\theta\mathcal{L}_{\mathrm{InfMasking}},
\qquad
g_{\mathrm{profile}}=\nabla_\theta\left(\alpha\mathcal{L}_{\mathrm{profile}}\right),
$$

并记录：

$$
\cos(g_{\mathrm{base}},g_{\mathrm{profile}}),
\qquad
\frac{\lVert g_{\mathrm{profile}}\rVert_2}
{\lVert g_{\mathrm{base}}\rVert_2}.
$$

诊断范围是 encoder、projection head 和全部共享参数，结果按 batch 汇总。负 cosine 表示该 batch 中两个目标对共享参数有直接方向冲突；范数比表示 profile 分支相对主损失的实际梯度强度。这个诊断不会单独证明 unique2 下降的原因，但可以判断是否有必要优先做 profile branch 的梯度协调。

### 6.1 V2 梯度诊断结果

我使用 G0 的三个 UniGIR checkpoint，各读取 8 个训练 batch，设置为 epoch 9/10，只做前向和两次反向求梯度，不调用 optimizer，不更新 prototype 或 queue。结果如下。

| model seed | 全参数 cosine 均值 | profile/base 范数比 | 冲突 batch 比例 | encoder cosine | head cosine |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.222 | 0.105 | 0/8 | 0.237 | 0.108 |
| 7 | 0.054 | 0.191 | 2/8 | 0.066 | -0.020 |
| 123 | 0.035 | 0.149 | 4/8 | 0.033 | 0.057 |

这个结果不支持 profile 梯度在所有 seed 上都和 InfMasking 主损失直接相反。它支持一个更谨慎的判断：profile 分支的梯度通常较小，平均约为主损失的 10%—19%；seed=7 和 seed=123 出现了部分 batch 的负 cosine，seed=7 的 projection head 平均方向略有冲突。因此，unique2 退化不能只归因于一个稳定的全局梯度冲突，后续应把重点放在表征几何和任务方向权衡上。

诊断输出保存在 `outputs/g0_gradconflict_s42_unigir_v1/`、`outputs/g0_gradconflict_s7_unigir_v1/` 和 `outputs/g0_gradconflict_s123_unigir_v1/`。状态检查均为 `state_unchanged=true`。这只是 24 个 batch 的局部证据，不能替代完整训练过程中的梯度统计。

### 6.2 V3 第一版代码状态

V3 的第一版代码骨架已经实现并完成 12 项远端单元测试：

- 新增 `losses/geodesic_profile.py`，构造对称 prototype k-NN 图；
- 使用 Floyd–Warshall 计算 prototype 间 shortest path；
- 使用 4 个最近 prototype 作为 anchors，近似样本到全部 prototype 的测地距离；
- geodesic target 仍使用 Sinkhorn，masked prediction 使用负距离 softmax；
- graph 从 detach prototype snapshot 构造，每个 epoch 最多刷新一次；
- 记录 graph 连通分量、不可达 pair、平均度和平均边权；
- 新增 `configs/model/unigir_geodesic.yaml`，但尚未启动 V3 训练。

G5 的第一轮 V3 短跑已经完成，但不能作为有效性能结果。graph 指标正常，然而 `loss_profile`、`profile_kl`、target confidence 和活跃 prototype 均为 0。原因是 geodesic distance 在 Sinkhorn 前直接指数化，早期 logits 发生浮点下溢。现已在指数化前做 row-wise max subtraction，并增加非零 profile 回归测试；修复后远端 3 项 geodesic 单元测试通过。独立运行名 `winpc_g5_v3_geodesic_s42_fix1` 已完成，修复版 profile 指标非零，最终性能比较和当前决策见本文顶部。

第二步是用 V3 进行短预算配对实验。第一轮只改 profile target 的距离计算，固定 V2 的 encoder、prototype 数量、EMA、queue、alpha、训练 seed 和训练预算。最小比较为：

| 版本 | 需要运行的任务 | 主要问题 |
|---|---|---|
| InfMasking | Baseline | 原始掩码协同学习是否有效 |
| UniGIR V2-Cosine | 已有 G0 结果 | 原型余弦轮廓是否带来正向信号及其代价 |
| UniGIR V3-Geodesic | 第一轮只做短预算配对实验 | 测地关系是否保留 synergy 收益并减轻 unique2 退化 |

只有当 V3 在相同协议下给出清晰的信息增益，才继续考虑 graph 更新频率、anchor 数量和 $k_g$ 的消融。

## 7. 停止条件

出现以下任一情况时，停止 V3 扩展并回到论文结论整理：

- prototype graph 大量不连通，且在固定 $k_g$ 下不能稳定修复；
- geodesic profile 的训练 loss 下降，但 synergy 和 unique2 都没有改善；
- V3 只改善单个 seed，不能复现 V2 的主要趋势；
- V3 需要同时改变 alpha、queue、EMA 或 encoder 才能得到结果，导致无法进行公平归因；
- 梯度诊断显示 profile 分支与主损失严重冲突，但没有先做最小协调实验。

当前不启动 100 epoch，也不同时搜索 graph、alpha、queue 和 prototype 数量。先运行固定协议的 V2/V3 短预算成对实验，再根据 synergy、unique2 和 graph 连通性共同判断是否扩展。
