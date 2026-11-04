# 636完整实验权重

[GitHub Release：full636-815cc37](https://github.com/xtr1976536/tokenizer-pde-discovery/releases/tag/full636-815cc37)。模型权重与恢复状态按SHA-256去重，40份唯一内容对应104个规范恢复路径，总计6,657,087,684字节（约6.20 GiB）。内容包括4份预训练模型、32组最终模型和4份独特恢复状态。`cycle_0.pt` 与共享预训练权重复用，`cycle_5.pt` 与 `searched.pt` 复用。

迁移备份和工作目录另外52个路径均逐份核对，其中仅第36 epoch恢复状态是新增内容，已单独保留一次。其余不再重复上传。小语料Release也按哈希比对过，没有与本次不同内容重复发布的权重。

查看下载计划：

```bash
python scripts/download_full636_weights.py
python scripts/download_full636_weights.py --arm ast_bpe_280 --case Wave_equation
```

下载并恢复指定模型/案例（同时恢复该模型的共享预训练权重）：

```bash
python scripts/download_full636_weights.py --arm ast_bpe_280 --case Wave_equation --download
```

完整恢复：

```bash
python scripts/download_full636_weights.py --download
```

下载器逐文件检查SHA-256及字节数，拒绝覆盖不匹配的现有文件；优先以硬链接恢复相同内容的多个路径。无需GitHub CLI。文件位于 `experiments/full/636/`，不会进入Git历史。

[weights_manifest.json](weights_manifest.json) 保存全部映射；[weights_upload_verification.json](weights_upload_verification.json) 保存GitHub服务器摘要复核。第1至4轮中间权重已在原实验完成时释放，Release不包含这些不存在的文件；保留情况见 [README](README.md#恢复及保留范围)。
