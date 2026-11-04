"""Run archived tests; optionally load MSVC-built unchanged native sources on Windows."""
import argparse,ctypes,importlib.util,os,sys
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--native-directory',type=Path);a=ap.parse_args();root=a.runtime.resolve()
 if not (root/'tests').is_dir():raise ValueError('Runtime tests missing')
 sys.path[:0]=[str(root),str(root/'src')]
 if a.native_directory:
  native=a.native_directory.resolve()
  import math_tokenizer.frontend_v6 as frontend
  import math_tokenizer.hard_ast_2026 as hard
  lib=ctypes.CDLL(str(native/'frontend.dll'))
  for name in ('mt_lex','mt_tree'):
   getattr(lib,name).argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];getattr(lib,name).restype=ctypes.c_int
  frontend._lib=lib
  spec=importlib.util.spec_from_file_location('_hard_ranked_2026',native/'_hard_ranked_2026.pyd');backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);hard._bridge=backend
 os.chdir(root)
 import pytest
 test_output=root.parent/('pytest-full636-'+root.name)
 return pytest.main(['-q','tests','--basetemp',str(test_output)])
if __name__=='__main__':raise SystemExit(main())
