# 636条语料审查

当前语料：[sources.jsonl](../../data/formulas/636/sources.jsonl)。原212条逐条保留，新增424条，其中23条替换了较重复的一阶实例，增加高阶、多维和非线性结构。

| 统计 | 替换前 | 当前 |
|---|---:|---:|
| 一阶EqWorld代表 | 146 | 123 |
| 直接来源方程（含记号规范化） | 90 | 108 |
| 不同无系数展开项 | 236 | 242 |
| 使用的原子token种类 | 34 | 36 |
| 最高空间导数四阶的记录 | 19 | 32 |
| 显式涉及三个空间方向的记录 | 6 | 8 |

全636条按空间坐标置换后的展开支撑统计为608种，新增424条互异且与原212条不重合。新23条中18条为直接来源或记号规范化，5条为同一非线性阻尼梁类别的明确函数选择。

[replacement_pairs.tsv](replacement_pairs.tsv)列出替换对应、公式和理由；[conversion_summary.json](conversion_summary.json)记录转换；[manifest.json](manifest.json)与round1/2/3.json保存三轮来源、编码、增强和嵌套关系检查。

当前增强与下游词表位于[data/formulas/636](../../data/formulas/636)。每条来源的位置、变换与SHA-256随源记录保存。早期数值词项问题已在公共评分层修正，见[评分验证](../scoring_fixes_validation/README.md)。
