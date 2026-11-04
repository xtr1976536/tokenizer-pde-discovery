# Tokenizer PDE Discovery

比较EqGPT GRE和Hard AST-BPE280/315/350在PDE结构发现上的表现。当前仅212条和636条两个嵌套语料规模；636采用最新23条多样性替换版，原212逐条保留。

## 实验设置

六层Transformer，d_model768，FFN2048，8头，d_k=d_v64；100epoch、batch128、Adam1e-4、clip1。位置容量50，沿用官方停止边界，最多48次新预测（BOS不计，实际EOS计入）。搜索5轮×400候选，每轮Top10继续训练5epoch、Adam1e-5。共同80/20抽样、串行生成后立即评分、相同清理/失败过滤/Top10/SFT，仅EqGPT使用SAT。原官方代码的明显运行/优化器绑定错误采用项目已检查的修正流程，因此不是原脚本逐行复现。

数据增强沿用第三套：每源32条加法项与标量乘法因子排列，分子分母不交换，EOS固定句尾，不截断。212与636分别学习Hard词表，故规模比较是词表学习和Transformer训练的整体流程效应。等epoch不等更新步数或算力。

## 安装和运行

Python3.11，建议独立虚拟环境。CPU/MPS/CUDA取决于Torch构建。

```bash
python -m pip install -r requirements-runtime.lock.txt
python -m pip install -e .
python -m scripts.pde_experiment plan
python -m scripts.pde_experiment prepare
python -m scripts.pde_experiment smoke --scale 212 --device mps
python -m scripts.pde_experiment smoke --scale 636 --device mps
python -m scripts.check_release --round 1
python -m scripts.check_release --round 2
python -m scripts.check_release --round 3
```

正式训练需显式执行：

```bash
python -m scripts.pde_experiment run --scale 212 --device mps
python -m scripts.pde_experiment run --scale 636 --device mps
```

现已完成212条小语料的100epoch正式实验，4个模型×8个案例共32组、64,000次候选尝试全部完成；636条未运行正式实验。完整日志、配置、轨迹和结果见 [正式实验报告](reports/full212_run/README.md)。以下为此前小试设置：小试保持六层架构，每份语料从完整源范围确定性选8条（636包含新增公式），预训练1epoch、batch2、Wave案例、2轮×4候选、微调1epoch。两规模四模型共8次训练、64次候选。无有效候选时跳过SFT，不额外补采样。三轮检查中的第一轮独立检查两个完整语料的全部增强，不能把小试误称全语料训练。

## 数据与文件

`data/formulas/212`：212源记录、6784增强及独立词表；`636`：636源记录、20352增强及独立词表。manifest绑定每个资产哈希。源码记录中的来源路径已随仓库复制，链接与SHA-256可核查；仅附本次所用来源快照，不包含大规模历史检索库。原始文献快照权利归原作者，保留其原始许可标识，本项目代码许可不覆盖文献。

`data/cases`与`data/mlp_cache`包含八案例数值输入和已拟合共享MLP。两模型/两语料复用同一数值环境，不自动重新拟合。固定缓存是条件比较，不声称重新采样所有随机因素。正式轮次不含最终PINN系数优化，不从结构命中推断系数精度。

`eqgpt_original`为官方固定代码及来源哈希；`src`包含当前共同实现；`native`包含Hard后端C源码。`reports/corpus_revision`保留23条替换及语料审查记录；`reports/checks`为新仓库当前三轮检查。历史报告提及的旧项目路径/编号仅作来源上下文，当前运行入口以本README为准。

`experiments/smoke`保留可复核日志与JSON轨迹，较大`.pt`检查点不入Git。重新运行小试会重建；不完整搜索不会被静默覆盖。依赖/协议改变时使用新输出目录并保留旧记录。第三轮清单不等于未来任意代码修改仍已审核。

## 已知限制

636记录对应608种当前展开支撑，原212的28条重复富余按要求保留；新增424全部互异且不与基底重合。原212条及其token ID保持不变。公共评分层`shared_scoring_fixes_v1`已修正`sint`、`uyyt`、复合导数和三维双拉普拉斯的求值，去重保留分子/分母角色，并移除按token组合置零的表示偏差；作者原始文件未改，历史错误与修正验证分别保留在`reports/scorer_audit`和`reports/scoring_fixes_validation`。人工审阅状态不伪造为通过。

GitHub repository: https://github.com/xtr1976536/tokenizer-pde-discovery (public).

上游分词器三指标迁移方案及当前212/636 benchmark的限制审查见 [reports/upstream_plan/README.md](reports/upstream_plan/README.md)。212完整实验权重将通过独立GitHub Release归档，避免将模型二进制写入Git历史。
