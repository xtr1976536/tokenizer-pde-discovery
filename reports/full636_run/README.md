# 636条下游完整实验

636条源公式增强为20,352条序列。EqGPT、Hard AST-BPE 280/315/350均完成6层预训练100 epoch；每模型15,900次优化器更新。8案例各搜索5轮，每轮400候选，Top10继续训练5 epoch，共32组、64,000次候选尝试。

| 模型 | Top-1正确 | Top-10包含正确方程 | 最后预训练loss |
|---|---:|---:|---:|
| EqGPT | 2/8 | 2/8 | 0.927088 |
| Hard 280 | 2/8 | 2/8 | 0.939876 |
| Hard 315 | 0/8 | 0/8 | 1.023522 |
| Hard 350 | 0/8 | 0/8 | 1.039134 |

最终一轮排名按原有 `evaluation.classify` 自动分类，已由归档检查重新计算；逐案例见 [summary.json](summary.json)。本轮只运行636条实验，未重新运行212条。两档实验沿用相同模型架构、评分、固定MLP数值缓存、搜索与微调预算；636条独立学习分词器词表。小语料结果见 [full212_run](../full212_run/README.md)。

设备为 NVIDIA RTX A500 Laptop GPU / CUDA，Python 3.11.18、PyTorch 2.14.1+cu130；实际环境见 [依赖冻结](runtime-requirements.freeze.txt) 与 [预运行检查](preflight636.json)。212条实验使用Apple MPS。相同epoch并不代表相同更新步数或算力，设备与独立词表也是比较条件，不能将结果差异单独归因于语料规模。当前流程不含最终PINN系数优化。

## 文件与完整性

- [experiments/full/636](../../experiments/full/636)：326份原始非权重文件，含4份预训练历史、32组完整候选、160份逐轮轨迹、诊断、机制分析、保留策略和完成标记。
- [artifacts_manifest.json](artifacts_manifest.json)：逐文件字节数及SHA-256。
- [run.log](run.log)：原始完整运行输出。
- [weights_manifest.json](weights_manifest.json)、[权重下载](WEIGHTS.md)：Release内容去重、原始路径恢复、恢复状态及保留范围。
- [runtime_source_manifest.json](runtime_source_manifest.json)、[runtime.patch](runtime.patch)：精确代码来源及仅有的运行增量。
- [orchestration_manifest.json](orchestration_manifest.json)、[orchestration](orchestration)：20份原始启动、预检查、迁移、保留、断点恢复与结果核对辅助脚本。脚本保留原机器路径用于溯源；可移植重跑使用下方独立重建入口，不直接运行这些历史包装脚本。
- [delivery_inventory.json](delivery_inventory.json)、[backup_deduplication.json](backup_deduplication.json)：归档选择、52个备份/恢复路径的哈希核对与重复内容处理。
- [三轮检查](AUDIT.md)：数据/产物、代码/结果、独立检出/远端交付。

## 代码复现

模型与算法基础代码已存在于提交 `815cc37adcdd714c9e9bd547a0561bb75f82901c`。大语料运行增量只有原子写入、逐epoch/每25候选断点、随机数与优化器/评估器状态恢复、输出位置和设备内存调度。增量作为补丁归档，完整74份Python/C源码按SHA-256校验；原始作者源码按既有提交复用。此次上传不改写 `src/`、`native/`、既有212条结果或上游实验。

在仓库根目录构造独立运行目录，避免覆盖已归档的结果：

```bash
python scripts/prepare_full636_runtime.py --output ../full636-runtime
cd ../full636-runtime
python -m pip install -r /path/to/tokenizer-pde-discovery/reports/full636_run/runtime-requirements.freeze.txt
export PYTHONPATH=.:src
export PYTHONUTF8=1
python -m pytest -q
# 需要开始新的完整实验时，显式指定新的空输出目录：
python -m scripts.pde_experiment run --scale 636 --device cuda --output ../full636-new-run
```

补丁还原后的源码指纹为 `e69f7045189d5b244ed84f562bf1246b4fd1312379aaa5a9a50f513029bd9c41`，入口脚本SHA-256为 `b8015ede38e76347ec4e14ff7f3b0c3ea08a74c1fff4ae07243ccef2440e5abc`，分别与原始 [RUN.json](../../experiments/full/636/RUN.json) 一致。Windows复核使用匹配Python架构的MSVC构建原始C前端与Hard编码扩展；Linux运行需要系统C编译器。以上重建仅处理新目录。

## 恢复及保留范围

运行曾在预训练和搜索中发生CUDA同步停滞，恢复记录见 [restarts636.json](restarts636.json)、第98 epoch与搜索恢复验证文件。原始候选前缀、随机数、优化器和评估器状态均被保留。训练期间的存储迁移未改变结果内容。

原实验在诊断与案例完成后释放了第1至4轮中间 `.pt`，128个中间权重路径现已不可恢复；各轮候选、loss、诊断及参数哈希仍完整保留。这是运行时已有保留策略，不是本次上传时删除。所有现存的预训练、最终权重和独特恢复状态均纳入Release；不会伪造已释放的文件。
