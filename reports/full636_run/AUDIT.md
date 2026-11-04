# 大语料归档的三轮检查

本次检查独立保存在 `reports/full636_run/`，不覆盖先前小语料与上游的三轮检查记录。

| 轮次 | 核对内容 | 记录 |
|---|---|---|
| 1 | 原始文件哈希、32组搜索、64,000候选、4份100 epoch历史、权重路径与已释放文件、备份去重 | [audit_round1.json](audit_round1.json) |
| 2 | 独立还原74份运行源码、原始指纹与入口哈希、数据清单、32组最终分类复算、模型/评分/分词代码一致性、47项测试 | [audit_round2.json](audit_round2.json)、[tests.log](tests.log) |
| 3 | 精确Git导出/独立检出、可重建性、保护既有目录、Markdown链接、GitHub Release每份资产大小与SHA-256、远端提交 | [audit_round3.json](audit_round3.json)、[remote_verification.json](remote_verification.json) |

可从仓库根目录重跑检查：

```bash
python scripts/audit_full636_archive.py --round 1
python scripts/prepare_full636_runtime.py --output ../full636-runtime
python scripts/audit_full636_archive.py --round 2 --runtime ../full636-runtime
python scripts/audit_full636_archive.py --round 3 --remote
```

测试使用原始C前端与Hard编码后端。此次不重新启动完整训练，不以重新生成的输出替换历史实验。权重的服务器验证仅在上传完成后记为通过。

额外交付证据：[模型参数指纹](weight_tensor_verification.json)核对36份实际模型状态与原始训练/搜索记录；[公开下载恢复](download_restore_verification.json)无凭据下载2份权重并校验4个硬链接路径；[打包备份内容](archive_content_verification.json)逐份哈希核对旧tar中的39份权重与8份已有MLP缓存，无遗漏的独特内容；[内容去重](content_deduplication.json)对照原仓库的Git对象。

Windows可使用[测试启动器](../../scripts/test_full636_runtime.py)加载MSVC编译的原始C源码：`python -X utf8 scripts/test_full636_runtime.py --runtime ../full636-runtime --native-directory /path/to/compiled-native`。目录中需要`frontend.dll`及匹配Python版本的`_hard_ranked_2026.pyd`；启动器仅适配动态库装载，不改变训练或断言。
