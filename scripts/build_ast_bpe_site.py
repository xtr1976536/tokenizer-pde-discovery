"""Rebuild the standalone export using only the Python standard library."""
from pathlib import Path
from html import escape
import json

root=Path(__file__).resolve().parents[1]
fragment=(root/'visualization/ast-bpe.fragment.html').read_text(encoding='utf-8')
trace=json.loads((root/'visualization/trace.json').read_text(encoding='utf-8'))
fragment=fragment.replace('__TRACE_DATA__',json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
assert '__TRACE_DATA__' not in fragment
shell=(root/'visualization/standalone-shell.html').read_text(encoding='utf-8')
assert shell.count('__AST_BPE_ESCAPED_FRAGMENT__')==1
page=shell.replace('__AST_BPE_ESCAPED_FRAGMENT__',escape(fragment))
(root/'docs/index.html').write_text(page,encoding='utf-8',newline='\n')
print('Built docs/index.html')
