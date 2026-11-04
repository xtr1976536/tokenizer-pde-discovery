# 训练覆盖与公平性检查

检查入口：[pde_experiment.py](../../scripts/pde_experiment.py)。[audit.json](audit.json)保存212/636两档完整增强语料的编码、拼批、词表范围、EOS与长度检查，以及八组模型最长输入的前向/反向结果。

| 源公式 | 每模型增强序列 | batch128每epoch更新 | 100epoch更新 |
|---|---:|---:|---:|
| 212 | 6784 | 53 | 5300 |
| 636 | 20352 | 159 | 15900 |

同一档四模型使用相同公式、增强顺序、主干、epoch/batch/优化器、MLP和搜索预算。Hard词表分别从各档未增强统一LaTeX学习。完整序列最大长度：GRE47、Hard280为47、315为43、350为39，均不超过50。

SAT仅EqGPT启用。词表大小与序列长度不同，因此等epoch不等预测token数或FLOPs。当前入口使用固定MLP与最小二乘评分，不执行最终PINN系数优化。

本目录记录训练准入检查；212条完整训练结果见[下游报告](../full212_run/README.md)，评分修正见[评分验证](../scoring_fixes_validation/README.md)。
