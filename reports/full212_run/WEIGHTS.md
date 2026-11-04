# 212完整实验权重

公开[GitHub Release](https://github.com/xtr1976536/tokenizer-pde-discovery/releases/tag/full212-815cc37)。164个唯一文件映射228个检查点路径，合计25,065,338,684字节（23.34 GiB）。检查点覆盖4个预训练模型、32组搜索的各轮续训及最终模型；重复硬链接路径只存一份。

权重清单：[weights_manifest.json](weights_manifest.json)。下载恢复脚本：[download_full212_weights.py](../../scripts/download_full212_weights.py)。在仓库根目录查看完整计划：

```bash
python scripts/download_full212_weights.py
```

按模型/案例恢复：

```bash
python scripts/download_full212_weights.py --arm ast_bpe_350 --case Wave_equation --download
```

全量恢复需要约24 GB空间和GitHub CLI。脚本逐个验证SHA-256，拒绝覆盖不匹配的本地文件，并用硬链接还原共享检查点路径。
