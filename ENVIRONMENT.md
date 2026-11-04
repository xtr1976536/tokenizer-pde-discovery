# 运行环境

Python 3.11；本地验证环境为macOS arm64与Apple MPS。requirements-runtime.lock.txt保存实际使用的直接依赖版本，pyproject.toml提供基础安装依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.lock.txt
python -m pip install -e .
python -m pytest -q
```

Hard AST-BPE首次运行时用系统C编译器构建native/，缓存写入.native_cache/。CPU、MPS、CUDA由所装PyTorch支持情况决定。下载Release权重需要GitHub CLI；其余实验入口见根README。

636条正式实验使用Linux / NVIDIA RTX A500 Laptop GPU、Python 3.11.18与PyTorch 2.14.1+cu130，实际冻结依赖保存在[full636_run](reports/full636_run/runtime-requirements.freeze.txt)。运行增量以补丁独立重建，不覆盖上面的既有环境；[复现和检查步骤](reports/full636_run/README.md)。636条权重下载器使用Python标准库，不需要GitHub CLI。Windows文本复核使用`python -X utf8`；原始C前端可用匹配Python架构的MSVC编译。
