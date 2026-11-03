import gzip,json
from scripts.pde_experiment import SETTINGS,verify

def test_formal_protocol_and_nested_scales():
 assert SETTINGS['layers']==6 and SETTINGS['epochs']==100
 assert SETTINGS['positions']==50 and SETTINGS['search_limit']==48
 assert SETTINGS['cycles']==5 and SETTINGS['attempts']==400 and SETTINGS['continuation_epochs']==5
 a,_=verify(212);b,_=verify(636)
 small=[json.loads(l) for l in (a/'sources.jsonl').read_text().splitlines()]
 large=[json.loads(l) for l in (b/'sources.jsonl').read_text().splitlines()]
 assert small==large[:212] and len(large)==636
