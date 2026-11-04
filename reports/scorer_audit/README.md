# 原评分实现审计

本目录保存修正前评分器的语义与边界检查。51个非控制词项共278个适用上下文记录：254通过、18不一致、6个拉普拉斯上下文不适用。错误集中于`sint`、`uyyt`、复合导数和三维双拉普拉斯。

同时发现乘/除项去重混淆，以及按token组合强制置零造成的表示偏差。公共层修正与当前验证见[scoring_fixes_validation](../scoring_fixes_validation/README.md)。

| 文件 | 内容 |
|---|---|
| atoms.json | 词项对照独立符号导数 |
| edge_cases.json | 去重、锚定、非有限行和退化列探针 |
| reward_oracle.json | 独立数值矩阵、最小二乘与reward核对 |
| real_targets.json | 八案例真实目标检查 |
| smoke_exposure.json | 64条历史小试候选涉及的词项 |
| tests.log | 当时测试输出 |

第一项锚定、非有限行过滤和按float32 reward去重的Top10规则仍沿用当前共同实现。
