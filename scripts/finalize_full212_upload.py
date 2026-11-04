"""Finish documentation and push only weight-delivery records after upload exits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/full212_run'


def run(*cmd):
    return subprocess.check_output(cmd, cwd=ROOT, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    while True:
        p = subprocess.run(['ps', '-p', str(args.pid), '-o', 'command='], capture_output=True, text=True)
        if p.returncode or 'scripts/upload_full212_weights.py' not in p.stdout:
            break
        time.sleep(30)
    verification = json.loads((REPORT / 'weights_upload_verification.json').read_text())
    assert verification['status'] == 'PASS'
    release = json.loads(run('gh', 'release', 'view', 'full212-815cc37', '--json', 'isDraft,url'))
    assert not release['isDraft']
    content = '''# 212完整实验权重

权重已上传并公开发布：[GitHub Release](https://github.com/xtr1976536/tokenizer-pde-discovery/releases/tag/full212-815cc37)。

164个唯一权重文件，映射回228个原始检查点路径，合计25,065,338,684字节（约23.34 GiB）。包含4个预训练模型、所有搜索轮次检查点及最终模型；共享的预训练与最终检查点通过路径映射恢复，不重复上传。

每个文件均核对本地SHA-256与GitHub服务器返回摘要及大小。见 [weights_manifest.json](weights_manifest.json)、[weights_upload_verification.json](weights_upload_verification.json)、[weights_upload.log](weights_upload.log)。

安装GitHub CLI后，在仓库根目录先查看下载计划：

```bash
python scripts/download_full212_weights.py
python scripts/download_full212_weights.py --arm ast_bpe_350 --case Wave_equation
```

添加 `--download` 执行下载。全量需要约24 GB额外空间；脚本验证哈希，通过硬链接恢复重复路径，拒绝覆盖已有但哈希不同的文件。权重位于Release，JSON轨迹与日志仍在Git主分支。
'''
    (REPORT / 'WEIGHTS.md').write_text(content)
    p = ROOT / 'README.md'
    text = p.read_text().replace('212完整实验权重将通过独立GitHub Release归档，避免将模型二进制写入Git历史。',
                                '212完整实验权重已通过独立GitHub Release归档，下载、路径恢复与哈希校验见 [权重说明](reports/full212_run/WEIGHTS.md)。')
    p.write_text(text)
    p = REPORT / 'README.md'
    text = p.read_text().replace('仅在本地保留的权重清单。约24 GB实际磁盘占用，按既有.gitignore未上传Git；硬链接导致逻辑大小求和更大。案例检查点哈希保存在各complete.json。',
                                '权重原始路径清单。全部权重现已上传独立Release，见 [权重下载说明](WEIGHTS.md)；硬链接导致逻辑大小求和更大。案例检查点哈希保存在各complete.json。')
    text = text.replace('克隆后可直接分析JSON。因权重未上传，不能仅凭归档文件加载本次训练模型或从检查点继续训练。',
                        '克隆后可直接分析JSON；按 [权重下载说明](WEIGHTS.md) 恢复Release权重后，可加载本次训练模型。搜索程序是否支持中途恢复仍由当前运行入口约束，下载权重不改变该逻辑。')
    p.write_text(text)
    p = REPORT / 'local_checkpoints.json'
    data = json.loads(p.read_text())
    for row in data['files']:
        row['availability'] = 'local_and_github_release'
    data['release'] = verification['release']
    p.write_text(json.dumps(data, indent=2) + '\n')
    paths = ['README.md', 'reports/full212_run/README.md', 'reports/full212_run/WEIGHTS.md',
             'reports/full212_run/local_checkpoints.json', 'reports/full212_run/weights_manifest.json',
             'reports/full212_run/weights_upload_verification.json', 'reports/full212_run/weights_upload.log']
    run('git', 'add', '--', *paths)
    run('git', 'diff', '--cached', '--check')
    print(run('git', 'commit', '-m', 'Publish verified full212 checkpoint delivery records', '--', *paths), flush=True)
    print(run('git', 'push', 'origin', 'main'), flush=True)
    status = dict(status='COMPLETE', release=release['url'], commit=run('git', 'rev-parse', 'HEAD').strip(),
                  finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (REPORT / 'weights_delivery_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
