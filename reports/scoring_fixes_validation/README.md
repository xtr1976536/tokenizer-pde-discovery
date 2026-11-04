# 公共评分修正与验证

评分版本：`shared_scoring_fixes_v1`。作者原始文件保持不变，修正在[scoring_fixes.py](../../src/math_tokenizer/eqgpt/scoring_fixes.py)接入。

- 修正`sint`、`uyyt`、`(u(u^2)xx)xx`和三维`BiLaplace(u)`的数值求值。
- 去重键分别保留分子/分母因子及乘法重数，区分`u*ux`与`u/ux`。
- 移除按token组合强制置零的表示偏差，保留共同数值矩阵、最小二乘、锚定和稀疏惩罚。

验证结果：44项测试通过；278个词项上下文中272通过、6个因空间维度不适用；八个真实目标reward与独立实现一致。等价`u²`与`u*u`探针评分一致。

64条历史小试候选在同CPU重评时，原版与修正版的状态和reward一致；直接比较旧MPS记录与新CPU结果时有一条约5.25e-6差异。[verification.json](verification.json)与same_device_replay64.json保留两种对照。

最终PINN未接入本次实验。第一项锚定、非有限行过滤和Top10的float32 reward去重规则保持现有设计。
