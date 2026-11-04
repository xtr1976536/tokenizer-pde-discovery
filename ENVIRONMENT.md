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
