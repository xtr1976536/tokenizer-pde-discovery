"""Pinned official reference tokenizers used in the current upstream study."""
import json
from tokenizers import Tokenizer
from math_tokenizer.baselines import HFTokenizer,TikToken
from math_tokenizer.baselines_v6 import Llama3Official
from math_tokenizer.corpus import ROOT

def load(method):
    if method == "qwen2.5":
        return HFTokenizer(Tokenizer.from_file(str(ROOT / "data/tokenizers/Qwen2.5-0.5B/qwen-tokenizer.json")))
    if method == "llama3":
        record = json.loads((ROOT / "data/tokenizers/llama3_provenance.json").read_text())
        return Llama3Official(record)
    if method == "o200k_base":
        return TikToken("o200k_base")
    raise ValueError(f"未知分词器：{method}")
