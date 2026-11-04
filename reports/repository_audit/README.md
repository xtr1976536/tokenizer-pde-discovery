# 仓库整理与三轮检查

本页保留此前小语料和上游的检查。新增636条正式实验的导入、结果复算、备份去重及Release检查单独记录于[full636_run/AUDIT.md](../full636_run/AUDIT.md)，不覆盖本页历史记录。

- cleanup_inventory.json：删除旧入口/过时文档、重复作者数据替代为链接的清单。
- duplicates.json：Git索引中完全相同文件的分组与处理。
- round1.json、round2.json、round3.json：每轮逐项验证结果。
- weights_release.json：GitHub Release资产、服务器摘要与本地检查点映射核对。

数据与相同结果通过相对链接共用。目录内`.model.json`描述符可能相同，但各自指向不同训练后端，保留其上下文。下游运行时代码快照保持full212的原始指纹，支持既有实验身份核对。

第三轮总是核对GitHub Release摘要；本地已下载的检查点会同时核对实际SHA-256，未下载时无需为检查额外恢复25 GB权重。

| 轮次 | 结果 | 核对内容 |
|---|---|---|
| [第一轮](round1.json) | PASS | 两档全量增强、来源/链接、作者哈希、32组下游与299份原始资产 |
| [第二轮](round2.json) | PASS | 44项测试、289份逐公式指标、共享函数与下游原始代码指纹 |
| [第三轮](round3.json) | PASS | 独立Git快照、44项测试、上游四项从头重跑、GitHub与本地164份权重 |
