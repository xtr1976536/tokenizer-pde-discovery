# 212条下游完整实验

代码提交：`815cc37adcdd714c9e9bd547a0561bb75f82901c`。设备：Apple MPS。运行从2026-11-03 18:06至20:41（Europe/Berlin），退出码0。

212条源公式增强为6784条序列。EqGPT、Hard AST-BPE 280/315/350均完成6层预训练100 epoch。8个案例各搜索5轮，每轮400候选，Top10继续训练5 epoch；共32组、64,000候选。位置容量50，最多48次新预测。评分版本：`shared_scoring_fixes_v1`。

| 模型 | Top-1正确 | Top-10包含正确方程 |
|---|---:|---:|
| EqGPT | 4/8 | 6/8 |
| Hard 280 | 1/8 | 1/8 |
| Hard 315 | 1/8 | 1/8 |
| Hard 350 | 4/8 | 5/8 |

由最终一轮排名和当前`evaluation.classify`自动分类。逐案例结果见[summary.json](summary.json)。

`experiments/full/212/`保存预训练loss、候选、逐轮搜索轨迹、诊断和完成哈希；`run.log`为完整输出；`artifacts_manifest.json`为SHA-256清单。权重见[Release下载说明](WEIGHTS.md)。
