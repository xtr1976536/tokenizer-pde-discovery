# 636条上游分词评估

四项实验：四分词器主比较、简单→复杂、词表容量、词表使用诊断。输入为636条未增强的统一GRE渲染LaTeX。

## 数据与指标

沿用旧实验seed20261028和来源/模板/规范化哈希连接分组，按8:1:1分配为509训练、64验证、63测试。共167组，最大237条；原212条在同一训练组内。精确ID与分组键见[protocol.json](protocol.json)、[input_records.json](input_records.json)。

| 指标 | 计算 |
|---|---|
| tokenizer compression | 初始词表实际编码token总数 / 最终词表实际编码token总数 |
| operation | 内部无token边界的完整LaTeX命令比例 |
| numbers | 内部无token边界的数字串比例 |

初始词表仅包含未合并的基础字节；编码不加BOS/EOS/PAD。每条公式保存initial_vocab_tokens与final_tokens。当前数字串全为单个数字，numbers恒为100%。

## 主实验

AST-BPE固定ascii98、总ID256，仅用509条训练公式重新学习。Qwen2.5、Llama 3、o200k_base保持旧实验官方词表。

| 方法 | 总ID | tokenizer compression | operation % | numbers % |
|---|---:|---:|---:|---:|
| Hard AST-BPE | 256 | 5.53529 | 100 | 100 |
| Qwen2.5 | 151665 | 1.79580 | 0 | 100 |
| Llama 3 | 128256 | 1.79580 | 0 | 100 |
| o200k_base | 200019 | 1.72502 | 0 | 100 |

主实验为固定四方法比较。operation为0表示命令内部存在切分边界。数据为统一渲染文本，官方基线保留各自词表容量。

## 其他三项实验

- **简单→复杂**：继承同一8:1:1划分，使用简单训练173条、复杂验证27条、复杂测试21条。复杂度沿用旧complexity-v2评分及30/40/30分位分层。
- **词表容量**：200/224/240/256/272/288/300/320/352/384/416/450共12档，保存训练、验证和测试结果；另有旧6基础表×22容量矩阵，132组合中100可运行、22训练覆盖不足、10容量不足。
- **词表使用**：24套容量/协议配置、7332条token记录，统计训练/验证/测试频率、未使用ID、低频与长token、来源数。

主实验和简单→复杂分别用各自训练子集学习词表；12档容量逐档独立训练。基础表矩阵使用各表512父词表的有序合并前缀。256档已核对与独立训练的词表及合并内容一致。同一冻结配置重复执行时复用保存的模型。

## 文件与复现

| 路径 | 内容 |
|---|---|
| [main](main)、[test_results.json](test_results.json) | 四方法主结果、AST-BPE256词表与训练来源 |
| [capacity](capacity) | 12档容量和alphabet_matrix基础表矩阵 |
| [simple_to_complex](simple_to_complex) | 复杂度迁移结果 |
| [vocabulary_usage](vocabulary_usage) | token频率、汇总与合并词项查看表 |
| [three_metrics.csv](three_metrics.csv) | 逐设置指标汇总 |
| [run.log](run.log)、[COMPLETE.json](COMPLETE.json) | 运行与完成记录 |
| [independent_audit.json](independent_audit.json) | 289份逐公式结果的独立重算 |
| [legacy_sources.json](legacy_sources.json)、[tokenizer_assets.json](tokenizer_assets.json) | 旧代码来源、共享函数与基线资产哈希 |

```bash
python -m scripts.run_current_upstream
```

共享词法计数位于scripts/upstream_metrics.py，基础表定义位于scripts/upstream_config.py，官方加载器位于scripts/upstream_tokenizers.py。完整仓库复现检查见[repository_audit](../../reports/repository_audit)。
