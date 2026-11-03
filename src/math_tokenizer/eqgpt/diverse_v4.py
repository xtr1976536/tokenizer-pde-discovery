"""Versioned four-scale experiment. Fail closed while the corpus is incomplete."""
import collections,gzip,hashlib,json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from .study3_corpus import SETTINGS as THIRD,paired_rows,SIZES,ARMS
from .paired_pipeline import POLICY,fingerprint
from .expanded_corpus import sha

DATA=ROOT/'data/formulas/diverse_pde_v4'
RUN=ROOT/'experiments/diverse_pde_v4'
REPORT=ROOT/'reports/diverse_pde_v4'
SCALES=(212,424,848,1696)
SETTINGS=dict(THIRD,layers=6,epochs=100,cycles=5,attempts=400,continuation_epochs=5,positions=50,search_limit=48,length_policy='official max_pos50; BOS included; while len(sentence)<max_pos-1',generation='paired_pipeline_v3; shared80/20; EqGPT SAT only')
QUALITY=dict(min_external_broad_families=25,max_external_broad_family_share=.10,
             max_instances_per_source_template=12,semantic_dedup=True,target_exclusion=True,
             fixed_gre_vocabulary=57,capacity=50,source_permutations=32)


def read_sources():return [json.loads(s) for s in (DATA/'sources_candidate.jsonl').read_text().splitlines()]

def code_hash():
    files=sorted([*ROOT.glob('scripts/*diverse*v4*.py'),*ROOT.glob('scripts/eqgpt/*diverse*v4*.py'),*ROOT.glob('scripts/*arxiv*v4*.py')])
    return hashlib.sha256((fingerprint()+json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},sort_keys=True)).encode()).hexdigest()


def require_ready():
    manifest=DATA/'prepared/manifest.json'
    if not manifest.exists():raise ValueError('Corpus incomplete/unreviewed: all four exact scales must pass data review before full training')
    m=json.loads(manifest.read_text())
    if m['scales']!=list(SCALES) or m['settings']!=SETTINGS:raise ValueError('Protocol mismatch')
    for name,h in m['files'].items():
        if sha(manifest.parent/name)!=h:raise ValueError('Prepared asset changed: '+name)
    if m.get('sources_sha256')!=sha(DATA/'sources_candidate.jsonl'):raise ValueError('Prepared source catalog changed')
    approved=DATA/'human_review.json'
    if not approved.exists():raise ValueError('Human formula review remains pending')
    review=json.loads(approved.read_text())
    if review.get('sources_sha256')!=sha(DATA/'sources_candidate.jsonl'):raise ValueError('Human review refers to different sources')
    ids={r['source_id'] for r in read_sources()}
    if review.get('approved_by')!='user' or set(review.get('passed_ids',[]))!=ids:raise ValueError('All source formulas require actual user review')
    checks=REPORT/'three_rounds.json'
    if not checks.exists():raise ValueError('Three current checks required')
    c=json.loads(checks.read_text())
    if c.get('code_sha256')!=code_hash() or c.get('prepared_sha256')!=sha(manifest) or len(c.get('rounds',[]))<3 or any(r['status']!='PASS' for r in c['rounds']):
        raise ValueError('Three current prepared-dataset checks required')
    return m


def candidate_tokenizers():
    return DATA/'candidate_tokenizers'/sha(DATA/'sources_candidate.jsonl')[:16]
