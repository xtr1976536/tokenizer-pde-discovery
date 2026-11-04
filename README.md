# Tokenizer PDE Discovery

数学分词器的上游评估与PDE结构发现实验。仓库包含源公式、分词器、训练与搜索代码、逐候选结果、运行日志和检查记录。

AST-BPE 动态演示：[在线版本（公开访问，无需登录）](https://ast-bpe-training-encoding-decoding.chenshuhang491.chatgpt.site)，xtr1976536 及其他访问者均可直接打开；也可下载并打开 [独立网站](docs/index.html)，逐步查看训练、编码和解码，以及每一步的顺序原因。[演示源码与复现说明](visualization/README.md)。公开 GitHub Pages 发布文件已就绪，管理员可在 Settings → Pages 选择 `main` 分支的 `/docs`。

初始基础词表由完整256字节表缩至98个ASCII/空白字符；计入3个控制符后，初始占用由259降至101个ID。构造与紧凑映射过程见[初始词表压缩说明](docs/initial_vocabulary.md)。

## 实验状态

| 实验 | 数据 | 状态 | 报告 |
|---|---|---|---|
| 上游分词评估 | 636条，训练/验证/测试509/64/63 | 四项实验完成 | [上游报告](experiments/upstream_current636/README.md) |
| 下游PDE发现 | 212条，增强后6784条 | 4模型×8案例完成，64,000次候选尝试 | [下游报告](reports/full212_run/README.md) |
| 大语料下游 | 636条，增强后20352条 | 4模型×8案例完成，64,000次候选尝试 | [大语料报告](reports/full636_run/README.md) |

## 上游：四个分词器

主实验固定Hard AST-BPE 256、Qwen2.5、Llama 3、o200k_base。AST-BPE仅使用509条训练公式学习词表；其余三种使用旧实验相同的官方固定词表。输入为未增强的统一GRE渲染LaTeX。

| 分词器 | tokenizer compression | operation (%) | numbers (%) |
|---|---:|---:|---:|
| Hard AST-BPE 256 | 5.5353 | 100 | 100 |
| Qwen2.5 | 1.7958 | 0 | 100 |
| Llama 3 | 1.7958 | 0 | 100 |
| o200k_base | 1.7250 | 0 | 100 |

- **tokenizer compression**：初始词表实际编码token总数 / 最终词表实际编码token总数。
- **operation**：内部未被token边界切开的完整LaTeX命令比例，如`\partial`、`\sin`。
- **numbers**：完整数字串比例。当前文本中的数字串均为单个数字，因此为100%。

同一8:1:1划分还用于简单→复杂、词表容量、词表使用诊断。其他容量与基础表的结果集中在容量实验中；主实验只有上述四种分词器。完整定义、词表训练记录与逐公式计数见[上游报告](experiments/upstream_current636/README.md)。

## 下游：PDE结构发现

比较EqGPT GRE与Hard AST-BPE 280/315/350，使用相同Transformer主干、数值缓存和评分流程，EqGPT保留SAT。

| 参数 | 设置 |
|---|---|
| Transformer | 6层，d_model768，FFN2048，8头，d_k=d_v64 |
| 预训练 | 100 epoch，batch128，Adam 1e-4，梯度裁剪1 |
| 序列容量 | 50位置，最多48次新预测 |
| 搜索 | 每案例5轮×400候选 |
| 每轮继续训练 | Top10，5 epoch，Adam 1e-5 |
| 数据增强 | 每源32条，交换加法项和标量乘法因子，EOS位于句尾 |

212条完整实验结果：

| 模型 | Top-1正确 | Top-10包含正确方程 |
|---|---:|---:|
| EqGPT | 4/8 | 6/8 |
| Hard 280 | 1/8 | 1/8 |
| Hard 315 | 1/8 | 1/8 |
| Hard 350 | 4/8 | 5/8 |

表中为自动结构分类。八案例共享已拟合MLP，当前流程不执行最终PINN系数优化。公共评分修正及回归验证见[评分记录](reports/scoring_fixes_validation/README.md)。

636条完整实验也已完成，相同模型与搜索协议的结果如下：

| 模型 | Top-1正确 | Top-10包含正确方程 |
|---|---:|---:|
| EqGPT | 2/8 | 2/8 |
| Hard 280 | 2/8 | 2/8 |
| Hard 315 | 0/8 | 0/8 |
| Hard 350 | 0/8 | 0/8 |

636条各模型预训练100 epoch、15,900次更新，使用CUDA；212条使用Apple MPS。两档分别学习词表，等epoch不等更新次数/算力，不能将差异单独归因于语料量。完整结果、运行增量、断点恢复和保留范围见[636条报告](reports/full636_run/README.md)。大语料运行源码通过基础提交加补丁单独还原，保留现有小语料及上游代码。

## 安装与运行

Python 3.11，需C编译器构建Hard后端；设备支持CPU、Apple MPS或CUDA。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.lock.txt
python -m pip install -e .

# 运行上游四项评估
python -m scripts.run_current_upstream

# 查看下游协议
python -m scripts.pde_experiment plan
```

仓库已附数据和词表，无需重新执行prepare。下游从头训练前，将已有`experiments/full/212`另存，再执行：

```bash
python -m scripts.pde_experiment run --scale 212 --device mps
```

小试入口为`python -m scripts.pde_experiment smoke --scale 212 --device mps`；从头运行同样需先另存历史`experiments/smoke/212`。已完成模型按下节恢复权重。

636条的原始断点恢复运行环境应独立还原，再指定新的输出目录；[复现步骤](reports/full636_run/README.md#代码复现)不会覆盖本仓库的历史实验。

## 权重

212条完整实验的164个唯一检查点约25.1 GB，对应228个路径，通过[GitHub Release](https://github.com/xtr1976536/tokenizer-pde-discovery/releases/tag/full212-815cc37)交付。进度与SHA-256验证见[权重说明](reports/full212_run/WEIGHTS.md)。

```bash
# 查看全量下载计划
python scripts/download_full212_weights.py
# 只恢复指定模型和案例；需要GitHub CLI
python scripts/download_full212_weights.py --arm ast_bpe_350 --case Wave_equation --download
```

636条权重通过[独立Release](https://github.com/xtr1976536/tokenizer-pde-discovery/releases/tag/full636-815cc37)交付：40份不同内容，约6.66 GB，对应104个规范路径，含全部现存预训练/最终模型和独特恢复状态。原实验已释放第1至4轮中间权重，其轨迹与哈希仍在；[保留说明及下载](reports/full636_run/WEIGHTS.md)。

```bash
python scripts/download_full636_weights.py
python scripts/download_full636_weights.py --arm ast_bpe_280 --case Wave_equation --download
```

## 文件导航

| 路径 | 内容 |
|---|---|
| [data/formulas](data/formulas) | 212/636源记录、32倍增强、下游词表、来源快照 |
| [data/tokenizers](data/tokenizers) | 上游官方基线词表及来源 |
| [data/cases](data/cases) | 八案例观测数据入口，链接到作者原始数据 |
| [data/mlp_cache](data/mlp_cache) | 八案例固定数值缓存 |
| [experiments/upstream_current636](experiments/upstream_current636) | 上游协议、词表、逐公式计数与诊断 |
| [experiments/full/212](experiments/full/212) | 下游训练历史、64,000候选、逐轮轨迹、完成哈希 |
| [experiments/full/636](experiments/full/636) | 大语料100 epoch历史、64,000候选、逐轮轨迹、诊断与完成哈希 |
| [reports/full636_run](reports/full636_run/README.md) | 大语料报告、运行补丁、去重清单、三轮归档检查与Release权重映射 |
| [experiments/smoke](experiments/smoke) | 两档历史小试记录 |
| [reports](reports/README.md) | 语料、评分、运行和三轮仓库检查索引 |
| [src](src) / [native](native) | 共享Python实现与C后端 |
| [eqgpt_original](eqgpt_original) | 固定版本作者源码、数据和来源哈希 |

636条保留原212条，新增424条互异结构；按空间坐标置换后的展开支撑统计，共608种结构。文献来源与23条替换记录见[语料审查](reports/corpus_revision/README.md)。

## 检查

```bash
python -m pytest -q
python -m scripts.audit_repository --round 1
python -m scripts.audit_repository --round 2
python -m scripts.audit_repository --round 3
```

三轮检查分别核对资产/链接与数据、代码/词表/结果、独立检出与GitHub权重交付。记录见[仓库检查](reports/repository_audit)。第三方源码、词表和文献的许可见[NOTICE.md](NOTICE.md)。

本次636条正式实验归档另做[三轮独立检查](reports/full636_run/AUDIT.md)，保留之前的小语料及上游检查记录。
