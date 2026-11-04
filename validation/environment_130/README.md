# H20 环境预检（212 / 636 条，130 epoch）

服务器工作目录：`/home/liugang/src/eq-bpe/project`，实际存储在 `/data1/users/liugang/src/eq-bpe/project`。正式运行器 `run_130.sh` 仅在预检 `READY.json` 存在、GPU 1 空闲且 `/data1` 至少剩余 100 GiB 时启动，212 与 636 两档顺序执行，结果分别写入 `runs/epochs130/212` 与 `runs/epochs130/636`。该脚本的默认 `--check` 只检查，不开始训练。

`preflight.py` 使用单卡运行全部样本检查、四模型在两档语料上的 CUDA 反向传播、八案例数值缓存复核，以及两档完整小规模流程。预检只写入本目录，原始数据和历史实验按哈希核对。服务器已保存 `READY.json`、测试日志、两档小试摘要和环境版本清单；小试权重在验证后已清理，以控制空间。

预训练 epoch 设置为 130，原协议中的 5 轮、每轮 400 个候选、每轮继续训练 5 epoch 等参数保持不变。服务器项目虚拟环境和 Python 头文件分别在 `.venv` 与 `.build-deps`；测试和正式运行均通过 `CPATH` 使用本地头文件。
