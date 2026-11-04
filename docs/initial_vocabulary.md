# 初始词表如何缩小

早期讨论中的1024是训练完成后的总词表预算。这里记录的是BPE学习合并前，初始词表占用的ID如何缩小。

## 从完整字节表到任务字符表

底层字节BPE训练器以全部256种单字节值为基础。PDE统一LaTeX文本采用ASCII记号，当前主配置将对外基础表固定为：

- 95个可打印ASCII字符：字节32–126；
- TAB、LF、CR三个空白字符：字节9、10、13。

因此，初始内容词表从256项缩至98项，回收158个单字节槽位。PAD、EOS、BOS另占3个ID：

| 初始组成 | 完整字节表 | 当前固定基础表 |
|---|---:|---:|
| 单字节内容项 | 256 | 98 |
| 控制符 | 3 | 3 |
| 初始占用总ID | 259 | 101 |

98是预先声明的字符集合。训练文本先检查字符覆盖，编码时遇到集合外的字节会明确报错。训练、验证、测试均核对解码还原。

## 取消额外的预留词项

当前Hard AST-BPE调用训练器时，设置`atom_budget=0`、`subtree_budget=0`。完整LaTeX命令和公式子树不额外占用初始ID；它们由后续受AST边界约束的BPE合并学习得到。因此，当前初始内容词表就是上述98个单字节项。

## 用紧凑映射暴露模型ID

`CompactAlphabet`保留底层训练器的256字节结构，对外执行三步：

1. 保留声明字符表内的单字节项，以及训练得到的多字节项。
2. 过滤声明表之外的单字节项，并把保留项映射为连续内容ID。
3. Transformer适配器添加3个控制ID，内容ID统一偏移3。

训练预算同时补偿内部保留、对外回收的字节槽位，使最终对外词表按指定总容量构建。合并规则与token字节内容沿用底层结果；新的ID映射在Transformer训练前固定。

当前上游主实验AST-BPE256，以及下游212/636两档的Hard280/315/350，均采用98字符基础表。更小基础表的对照保留在上游词表容量实验中。

## 实现位置

- [HardASTBPE.train](../src/math_tokenizer/hard_ast_2026.py)：关闭命令/子树预留，使用AST约束合并。
- [CompactAlphabet与train_compact](../src/math_tokenizer/eqgpt/compact_alphabet.py)：覆盖检查、过滤单字节项、连续ID映射和预算补偿。
- [LatexAdapter](../src/math_tokenizer/eqgpt/latex_tokenizers.py)：PAD/EOS/BOS及内容ID偏移。
- [下游基础表](../src/math_tokenizer/eqgpt/eqgpt_fast_config.py)、[上游基础表配置](../scripts/upstream_config.py)：声明98字符及其他对照字符表。
