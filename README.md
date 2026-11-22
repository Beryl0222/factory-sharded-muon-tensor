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
