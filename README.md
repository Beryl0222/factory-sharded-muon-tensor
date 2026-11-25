# 矩阵分片的 Muon 正交化与权重交接

训练台包含非方阵和零矩阵两种逻辑投影权重，分片选择由运行方给出。 数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

`matrices` 提供原矩阵形状及初始代次。分片区间和轴使用原始行列坐标。物理布局 `RC` 表示行、列，`CR` 表示列、行。`v=βv_old+(1−β)g` 为新动量，更新核使用 v。按逻辑长轴为列解释迭代，行数大于列数的矩阵应先转置，最后转回。除以 v 的 Frobenius 范数，零矩阵更新为零且不除零。每轮以当前 X 做三项多项式 `a X + b (XXᵀ) X + c (XXᵀ)^2 X`。参数更新为 `w_new=w_old−lr*X_final`，动量与参数持久值为 float32。精度契约比较回传的最终 X 与 float64 的同迭代定义，容差为最大绝对元素误差。若更新最终 Frobenius 范数超过组的允许值，不对其缩放后冒充合规。推理副本是 float32 参数输入 round-to-nearest ties-to-even 的 binary16 表示。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
```

现有检查核对采集格式和所附数据的结构。

---

## 分片 Muon 实现（`shard_muon/`）

只依赖 numpy。提供纯函数式分片核、精度门，以及有状态的训练协调端
（上报 → 成功矩阵/分片更新/延迟原因 → 幂等提交确认/回执查询）。

```sh
# 完整回放 capture 的两批上报、提交、重发确认、列分片等价资料
python -m shard_muon replay
python replay.py                 # 等价入口
python -m shard_muon replay --state-dir /tmp/muon_state --values

# 跨进程按幂等键（或 step，取最近回执）查询当时冻结的确认信息
python -m shard_muon receipt /tmp/muon_state 'train-9#confirm-2'

# 容量估算；--run 会按 capacity.json 的种子真跑 18432x432 行/列分片核对
python -m shard_muon capacity
python -m shard_muon capacity --run

# 全部测试（含上述故事的可执行规格）
python -m unittest discover -s tests -v
SHARD_MUON_RUN_CAPACITY=1 python -m unittest discover -s tests -v
```

根目录另外放了 `muon.py / sharded_muon.py / muon_shard.py / muon_service.py /
coordinator.py / service.py / trainer.py / solution.py / main.py` 等兼容入口，
全部 re-export 同一个 `shard_muon` 包；`python solution.py replay` 也可运行。

### 30 秒上手

```python
import numpy as np
from shard_muon import (MuonCoordinator, MatrixSpec, IterationSpec,
                        ShardPiece, sharded_muon, assemble_logical_x)

# 1) 有状态训练端（records 可直接传 data/capture.json 的 records 字典）
coord = MuonCoordinator(
    {"Q": MatrixSpec("Q", shape=(7, 3), generation=12, error=1e-4, max_norm=2.3)},
    iteration_spec=IterationSpec(iterations=4, beta=0.9, lr=0.004),
)
coord.initialize("Q", weight=w32, momentum=v32, generation=12)

rep = coord.report("train-9", submissions, tensors)   # capture 风格字典 + tensors
rep.newly_successful          # 本次新算好的逻辑矩阵
coord.delay_reasons("train-9")  # 未就绪矩阵的延迟原因
q = rep.deliveries["Q"]
q.selected_precision          # 'binary16' | 'float32' | 'float64'
q.logical_x                   # 完整逻辑 X（训练/交接侧核对用）
q.shard_updates               # 每片 x/delta/新权重/新动量/推理副本 + 物理朝向
q.physical_update_map()       # {partition: 物理布局的 x_physical}

rc = coord.commit("train-9", key="train-9#confirm-2")  # 原子生效
again = coord.commit("train-9", key="train-9#confirm-2")  # 冻结回执重发，不二次应用
coord.get_confirmation("train-9#confirm-2")               # 也可直接传 step
# rc.iteration_spec / rc.actual_precision / rc.waiting /
# rc.older_steps_waiting / rc.newly_applied_matrices

# 2) 纯分片核：一套完整覆盖的动量块 -> 各块最终 X
pieces = [ShardPiece("q0", axis=0, lo=0, hi=3, logical=v[0:3, :], layout="RC"), ...]
result = sharded_muon((7, 3), pieces, IterationSpec(4), mode="float64")
X = assemble_logical_x((7, 3), result)
```

也可直接跑采集故事：`play = shard_muon.play_capture("data/capture.json")`。

### 每条题目要求对应的位置

| 要求 | 实现 / 核对 |
| --- | --- |
| 逻辑矩阵、原始行列坐标、物理 RC/CR 轴序 | `contract.MatrixSpec`；`layout.unpack_physical/pack_physical`；区间永远是原始逻辑坐标、左闭右开 |
| 可沿行或列切分 | `coverage.analyze_coverage` 分别建 axis=0/1 覆盖；`engine.sharded_muon` 两种分块代数 |
| 同一步只认一套完整覆盖，不把两套叠起来计数 | `coverage.choose_coverage`（默认优先 axis=0），另一套只进 `ignored_complete_axes`；测试 `test_two_complete_coverages_not_stacked` |
| 梯度+旧动量换到逻辑身份再更新 | `service._evaluate_matrix_step`：unpack 梯度、按坐标切旧动量，`v=βv_old+(1−β)g` |
| 公开迭代 + 转置契约 | `reference.muon_reference`（float64 oracle）与 `engine.sharded_muon`；`transpose = rows > cols` **只看完整逻辑形状**，不看本地分片形状（有专门回归测试） |
| 薄维放得下全局 Gram、放不下整份高维副本 | Gram 仅 `thin²`；引擎全程分块；`capacity.scratch_plan`：18432×432 的 f64 整副本 ≈ 60.8 MiB > 6 MiB，Gram ≈ 1.42 MiB |
| Gram 每个有效元素只算一次 | X 行块时只算无序块对 `i≤j`（`k(k+1)/2` 对/轮）并镜像；列块时每个不相交坐标块贡献一次；`result.trace["gram_block_pairs"]` 可查 |
| 应用迭代计算跨切分轴贡献 | X 行块：`Y_i = aX_i + Σ_k B_ik X_k` 显式跨片求和；X 列块：全局 `B X_i`（B 已含所有片的 Gram 贡献） |
| 每轮重新形成 Gram | 迭代循环内每轮用当前 X 重算 `A=XXᵀ`（初版原型曾把 Gram 放循环外，已被测试钉住） |
| 全零是可交付分支 | 范数精确为 0 → 全零 X、不做除法；任何精度都通过并正常提交（capture 的 R → binary16） |
| 损坏/缺席不能当零 | NaN/Inf→`corrupt_shard`；张量缺失/`None`→`absent_shard`；形状/轴序不符→`corrupt_shard(bad_shape)`；旧代次→`version_mismatch`；缺口→`missing_coverage`。绝不拿零填洞 |
| 低精度候选 + 误差声明 + 范数门 | `engine.evaluate_matrix`：binary16→float32→float64，比较实际回传 X 与 float64 同迭代定义的最大绝对元素误差；`||X_final||_F > max_norm` 判 `norm_exceeded`，**不缩放** |
| 小矩阵/近秩亏/不同分片排序同一口径 | 全部走同一 oracle + max-abs 口径；坐标排序使结果与到达顺序位级一致；近秩亏严公差会自动升到 float64 |
| 一块不合格→整组换精度；仍不过保留旧 w/v | 单矩阵走完整升级梯；显式多矩阵组用 `_group_mode_for_step` 取全组统一精度并留 `group_precision_audit`；范数等终局失败时不提交，旧权重/动量/代次不变 |
| w、动量、binary16 推理副本一起生效；缺片无新矩阵 | `MatrixDelivery` 先构建完整 `weight_new/momentum_new/inference_new`，`commit` 时整体替换；覆盖不完整则没有 delivery |
| 矩阵组之间无交付依赖 | 逐矩阵独立评估/提交；capture 首批 Q 等待不阻塞 R |
| 上报另一批梯度→成功矩阵、分片更新、延迟原因 | `MuonCoordinator.report` 支持同一步多批；`newly_successful / shard_updates / delay_reasons` |
| 提交确认重发可查规格/实际精度/旧步等待 | `commit(key=...)` 幂等；`get_confirmation(key_or_step)` 返回冻结的 `iteration_spec / actual_precision / waiting / older_steps_waiting`；可选 `state_dir` 落 JSON+npz，新进程 `MuonCoordinator.load(dir)` 后仍可查 |
| 同一步更新不能应用两次 | `(step, matrix)` applied 集合硬保护；同键重发返回原回执（`resend_of_original=True`）；无键重复提交回放最近回执；代次只 +1 |

### capture 数据上的实际结论

* 首批 `q0(v12)+q1(v11 旧代次)+q2(v12)+r0(v44)`：**R 成功**（全零 →
  `binary16`，`||X||=0`），**Q 等待**：`version_mismatch(q1, 11≠12)` 与
  `missing_coverage [3,6)`。提交 `train-9#confirm-1` 只应用 R（代次 44→45）。
* 补报 `q1(v12)` 后 Q 完整：`binary16` 候选最大误差 ≈ **1.32e-2 > 1e-4**
  被拒，自动升 `float32`，最大误差 ≈ **9.7e-7**，`||X||_F≈1.092 < 2.3`
  通过；提交 `train-9#confirm-2` 只新应用 Q（代次 12→13），R 不二次应用。
* `alternative_submissions`（列切分 + CR）在全新协调端回放，与行切分结果
  最大差 ≈ **1.6e-6**（同为 float32，同一比较口径）。
* 持久值：权重/动量 `float32`，推理副本为 `weight.astype(np.float16)`
  （numpy 即 round-to-nearest-ties-to-even 的 binary16）。
* 容量种子回放（seed 852，18432×432，23 片，5 轮）：行/列分片与全矩阵
  float64 oracle 互差 ~1e-16，3–4 秒跑完。

### 设计边界

协调端（训练/交接侧）会装配完整逻辑数组做权威误差比对与原子交接；
**板端/容量路径**（`shard_muon.engine.sharded_muon` 与
`capacity.run_capacity_replay`）只持有薄维 Gram 与分片工作块，不分配整份
高维 float64 副本。`capacity.scratch_plan` 同时给出主要暂存口径与含
matmul 第二输出块的保守口径：该规模下 binary16/float32 保守峰值在
6 MiB 内，float64 主要缓冲在预算内、保守双缓冲约 10 MiB。
