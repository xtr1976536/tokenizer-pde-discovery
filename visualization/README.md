# AST-BPE 交互演示

`../docs/index.html` 是导出的完整静态网站，可直接用浏览器打开，无需 Python、服务器 API 或 Node 依赖。GitHub Pages 可选 `main` 分支的 `/docs` 文件夹发布。

`ast-bpe.fragment.html` 保存可编辑演示源码；`trace.json` 保存由仓库算法生成并校验的训练、编码轨迹。网站内的“为什么先做这一步”随当前阶段和步骤更新。

本演示用四条明确列出的教学语料、内部容量 272（256 单字节 + 16 个学习 token），展示底层 HardASTBPE。不是 212/636 条正式实验结果，也不是紧凑映射后的模型 ID。算法来源固定在提交 `d1eda7638237dbec3ce4474e25d24d70820cab6e`；创建网站时，当前 `a1d5ac6` 的相关算法文件与该版本一致。

## 更新轨迹

在已安装仓库 Python 依赖及 C 编译器的 Linux/macOS 环境执行：

```sh
python scripts/trace_ast_bpe_demo.py
python scripts/build_ast_bpe_site.py
python -m http.server 8000 --directory docs
```

Windows 可预编译原始 `native/frontend_v6.c`（导出 mt_lex、mt_tree）与 `native/hard_ranked_2026.c`（Python 扩展），将 `frontend.dll` 和 `_hard_ranked_2026.pyd` 放在 `.native_cache`，或通过 `AST_BPE_NATIVE_DIR` 指定目录。编译时须匹配当前 Python 版本与架构。

轨迹生成器调用原始 `HardASTBPE.train`，重放每轮训练验证最高频及同频排序。每次编码追踪同时对照 Python 参考编码和 C 原生编码，并检查逐字节还原；包括演示语料和仓库 212 条源公式。浏览器只播放这些已记录轨迹，不在网页中另写一个近似训练器。

## 更新页面

```sh
python scripts/build_ast_bpe_site.py
```

构建脚本使用初次导出保留的独立展示外壳 `standalone-shell.html`，把最新 fragment 和 trace 注入 `docs/index.html`。生成文件应随源码一起提交；常规访问不需要构建。网站仅使用浏览器本地存储恢复步骤，不上传用户数据。

## 为什么按这个顺序

- 训练：先取得源区间约束，才能统计合法 pair；先选当前频次最大者，再按数字 pair ID 排同频项；一次合并后邻接关系变化，必须重新计数。
- 编码：词表与 rank 已冻结；按 rank、左端位置处理当前已学且相邻的候选；执行前检查有效性与区间合法性，再更新左右邻居。
- 解码：按输入 ID 顺序取字节并拼接，最后统一 UTF-8 解码。无需 AST，也无需逆向重放合并规则。
- 本例先合并 `u_`，是因为合法频次为 5，与 `v_` 同频但 `(117,95)` 小于 `(118,95)`；不是因为字母 u 在数学上更重要。

验证范围：217 次 Python/C/还原检查、16 轮训练选择；桌面和窄屏的三阶段交互、边界检查与步骤解释。分词器生产代码未修改。
