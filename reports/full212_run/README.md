# 212条小语料完整实验

运行代码提交：`815cc37adcdd714c9e9bd547a0561bb75f82901c`。本地Apple MPS运行，仅212条，636条未启动。

UTC开始：2026-11-03T17:06:18.947728+00:00；结束：2026-11-03T19:41:12.786540+00:00。退出码0，状态 `COMPLETE_VERIFIED`。

## 配置与结果

212条源公式，增强后6784条序列。4个模型均为6层、预训练100 epoch；8个案例各搜索5轮，每轮400候选，Top10继续训练5 epoch。共32组、64,000次候选尝试。位置容量50，最多48次新预测。评分层 `shared_scoring_fixes_v1`。完整配置及代码/数据指纹见 [RUN.json](../../experiments/full/212/RUN.json)。

| 模型 | Top-1正确 | Top-10包含正确方程 |
|---|---:|---:|
| eqgpt_reference | 4/8 | 6/8 |
| ast_bpe_280 | 1/8 | 1/8 |
| ast_bpe_315 | 1/8 | 1/8 |
| ast_bpe_350 | 4/8 | 5/8 |

按最终一轮候选排名，由当前 `evaluation.classify` 自动分类；冗余结构不计为严格正确。尚未逐条人工复核，也不是多随机种子统计。

## 归档文件

- [run.log](run.log)：完整原始标准输出及错误输出，包含进度控制字符。
- [summary.json](summary.json)：32组分类及聚合结果。
- [launch.json](launch.json)、[launch.py](launch.py)、[exit.json](exit.json)：启动信息、监督脚本、退出状态。
- [experiments/full/212](../../experiments/full/212)：全部294份JSON，包括预训练loss历史、全部候选尝试、逐轮轨迹、诊断、机制、配置、完成标记和文件哈希。
- [artifacts_manifest.json](artifacts_manifest.json)：原始实验文件与运行记录的大小和SHA-256清单。
- [local_checkpoints.json](local_checkpoints.json)：仅在本地保留的权重清单。约24 GB实际磁盘占用，按既有.gitignore未上传Git；硬链接导致逻辑大小求和更大。案例检查点哈希保存在各complete.json。

监督脚本在运行结束时验证全部32组完成标记及文件哈希、每组2000次候选与5轮记录，并确认共享MLP未变。提交前再次验证非权重案例文件哈希及4个模型100 epoch记录。代码、数据和权重未修改。

克隆后可直接分析JSON。因权重未上传，不能仅凭归档文件加载本次训练模型或从检查点继续训练。
