# UniGIR 动机与行动记录

本文档记录研究过程中遇到的问题、形成的判断、对应的代码修改和实验结果。记录重点放在为什么做出改变，以及改变之后是否得到证据。

## 2026-09-10：从短训练结果重新检查方法

### 遇到的问题

UniGIR V2 已经可以正常训练，原型预测指标也在上升。但在 R-U 和 S 两组短训练中，UniGIR 的下游平均 acc@1 都低于 InfMasking Baseline：R-U 组为 0.355 对 0.372，S 组为 0.345 对 0.375。

这说明当前版本虽然产生了学习信号，但还没有证明全局轮廓目标能改善下游任务。继续直接增加训练 epoch 会消耗大量 CPU 时间，却不能排除实现或目标设计本身的问题。

### 重新检查后的判断

当前实现有两个需要先澄清的地方：

1. Sinkhorn 分配主要使用当前 batch 的完整视图。EMA 只更新原型向量，并没有让轮廓计算看到最近 batch 的特征。因此当前轮廓更接近 batch 内的均衡分配，和 idea 中的全局关系轮廓还有差距。
2. profile loss 原先和其他损失一起放入列表后求平均。profile loss 的实际影响会随损失项数量变化，不能直接解释为一个明确的权重。

### 采取的改变

#### 1. 加入跨 batch feature queue

在 `PrototypeAlignment` 中加入可选的 feature queue。训练时，当前 batch 的完整视图与 queue 中最近保存的完整视图一起参与 Sinkhorn 分配，但只取当前 batch 对应的分配作为伪标签。这样可以在不保存全量数据的情况下，让原型分配使用更多历史特征。

queue 采用环形写入，默认长度为 1024。验证阶段不更新 queue，避免验证数据改变训练状态。

#### 2. 把 profile loss 改成显式权重

总损失改为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{InfMasking}}
+
\alpha\mathcal{L}_{\mathrm{profile}}.
$$

其中，`alpha` 由 `profile_kwargs.loss_weight` 控制，当前配置为 $\alpha=0.25$。原始 InfMasking 关闭 profile 分支时，损失计算保持不变。

#### 3. 增加诊断指标

当前训练额外记录：

- `loss_base`：原始 InfMasking 损失；
- `loss_profile`：乘以 $\alpha$ 后的 profile loss；
- `profile_kl`：完整视图轮廓和掩码视图预测轮廓之间的 KL 散度；
- `profile_usage_entropy`：原型使用分布的归一化熵；
- `profile_active_prototypes`：当前 batch 中有明显使用量的原型数；
- `ssl_acc_profile`：掩码视图预测完整视图原型的准确率。

这些指标用来区分原型塌缩、目标过弱、目标过强和目标与下游任务无关等情况。

#### 4. 增加低成本训练开关

在 `train_trifeatures.yaml` 中增加 `enable_linear_probe`。设置为 `false` 时，训练过程中不建立四个下游任务的 probing callback，只保留自监督训练和诊断指标。这样可以用 CPU 快速完成权重、原型数和目标形式的初步比较，最后再对少数候选模型做 probing。

### 已完成的代码修改

- `PrototypeAlignment` 已加入可选的跨 batch queue，默认长度在 UniGIR 配置中设为 1024；
- EMA 原型更新仍然只在训练阶段执行；
- profile loss 已改为显式加入总损失，形式为 $\mathcal{L}=\mathcal{L}_{\mathrm{InfMasking}}+\alpha\mathcal{L}_{\mathrm{profile}}$，当前 $\alpha=0.25$；
- 已加入 `profile_kl`、`profile_usage_entropy` 和 `profile_active_prototypes` 等诊断量；
- `enable_linear_probe=false` 可以关闭训练过程中的下游 probing。
- 已准备 `../run_scripts/cpu_diagnostics.ps1`，按单进程顺序运行 baseline、queue、$K$、$\alpha$ 和双向预测的低成本对照。

### 当前验证状态

代码差异检查已通过。运行时冒烟测试暂未完成：当前 Windows 环境无法创建新的 Python 子进程，错误发生在启动解释器阶段，尚未进入项目代码。待系统恢复进程创建后，先运行 queue、EMA 和反向传播测试，再进行 1 至 2 个 epoch 的 CPU 诊断实验。

低成本诊断脚本已经准备好，所有实验都关闭 probing，每个实验使用 `max_size=512`、`embed_dim=256`、2 个 epoch，并且严格串行运行，避免再次出现多个 Python 进程互相抢占 CPU 的问题。

### 当前实现位置

- 跨 batch queue、原型统计：`../losses/prototype_alignment.py`；
- 显式 profile 权重和诊断日志：`../losses/infmasking_loss.py`；
- UniGIR 默认配置：`../configs/model/unigir.yaml`；
- 低成本训练开关：`../main_trifeatures.py`、`../configs/train_trifeatures.yaml`。

### 验证计划

先进行代码级检查，确认 queue、EMA 和反向传播没有数值问题。之后在 CPU 上使用以下设置做小规模诊断：

- `max_size=512或1024`；
- `embed_dim=256`；
- `num_mask=2或3`；
- 训练 1至2 个 epoch；
- `enable_linear_probe=false`；
- 比较 `alpha`、$K$ 和单向/双向轮廓预测。

优先观察 profile loss、profile KL、原型使用熵和 active prototypes。只有当这些指标正常，且不同设置之间有稳定差异，才继续做下游 probing。

### 当前阶段的目标

当前阶段不追求在 CPU 上完成论文规模训练，而是回答三个实现问题：

1. 加入历史特征后，轮廓是否比单 batch 分配更稳定；
2. 哪个 profile loss 权重能让原始任务保持正常训练，同时让新目标产生可观察的影响；
3. 原型轮廓是否包含与 share、unique 和 synergy 有关的信息。

如果低成本诊断不能回答这些问题，就不继续增加训练时间，而是重新检查 profile 目标是否与 InfMasking 原有损失重复。
