"""Audit every formal636 artifact before emitting the final result summary."""
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent.parent
dest=root/'outputs/large636'
arms=('eqgpt_reference','ast_bpe_280','ast_bpe_315','ast_bpe_350')
cases=('Wave_equation','Convection_diffusion_equation','KG_equation','Burgers_equation',
       'Eq_6_2_12','PDE_divide','KdV_equation','Chaffee_Infante_equation')
complete=json.loads((dest/'COMPLETE.json').read_text())
run=json.loads((dest/'RUN.json').read_text())
assert complete['status']=='PASS' and complete['scale']==636 and complete['smoke'] is False
assert run['scale']==636 and run['smoke'] is False and run['device']=='cuda'
assert run['settings']['epochs']==100 and run['settings']['layers']==6
assert run['settings']['cycles']==5 and run['settings']['attempts']==400
cache={}
def digest(path):
    stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    if key not in cache:
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        cache[key]=h.hexdigest()
    return cache[key]

summary=dict(status='PASS',scale=636,sources=636,augmented_samples=20352,
             pretraining_epochs_per_model=100,optimizer_updates_per_model=15900,
             candidates_per_case_model=2000,total_candidates=64000,total_case_models=32,
             small_corpus_experiment_run=False,run_identity=run,models={})
for arm in arms:
    training=json.loads((dest/'shared_pretrain'/arm/'training.json').read_text())
    assert training['epochs']==100 and training['batch_size']==128 and training['samples']==20352
    assert training['steps_per_epoch']==159 and training['optimizer_steps']==15900
    assert len(training['history'])==100 and training['initial_sha256']!=training['final_sha256']
    assert all(math.isfinite(row['loss']) for row in training['history'])
    results={}
    for case in cases:
        folder=dest/'cases'/case/arm
        marker=json.loads((folder/'complete.json').read_text())
        for name,expected in marker['files'].items():
            assert digest(folder/name)==expected,(case,arm,name)
        search=json.loads((folder/'search.json').read_text())
        assert search['settings']==run['settings'] and search['diagnostics_status']=='complete'
        assert len(search['attempts'])==2000 and len(search['history'])==5 and len(search['checkpoints'])==6
        hardware_errors=[r.get('error') for r in search['attempts'] if any(
            token in r.get('error','').lower() for token in
            ('out of memory','cuda error','device-side assert','cublas_status','cudnn_status'))]
        assert not hardware_errors,(case,arm,hardware_errors[:3])
        assert search['checkpoints'][0]['sha256']==training['final_sha256']
        assert search['checkpoints'][-1]['sha256']==search['history'][-1]['after_sha256']
        for cycle in range(1,6):
            rows=[r for r in search['attempts'] if r['cycle']==cycle]
            assert len(rows)==400 and {r['attempt'] for r in rows}==set(range(400))
            h=search['history'][cycle-1]
            assert all(math.isfinite(v) for v in h['continuation_losses'])
            assert len(h['continuation_losses'])==(5 if h['continuation_sequences'] else 0)
            if h['continuation_sequences']:assert h['before_sha256']!=h['after_sha256']
        mechanism=json.loads((folder/'mechanism.json').read_text())
        diagnostics=json.loads((folder/'diagnostics.json').read_text())
        assert diagnostics['truth_not_injected'] and diagnostics['executed_after_search']
        results[case]={k:mechanism[k] for k in ('top1_correct','top10_correct','evaluated',
                       'ever_raw_complete_correct','ever_effective_correct_scored','invalid_status_counts')}
        results[case]['final_equations']=search['history'][-1]['equations']
        results[case]['final_rewards']=search['history'][-1]['rewards']
        results[case]['hardware_failure_count']=len(hardware_errors)
    summary['models'][arm]=dict(cases=results,top1_correct=sum(r['top1_correct'] for r in results.values()),
        top10_correct=sum(r['top10_correct'] for r in results.values()),
        first_pretrain_loss=training['history'][0]['loss'],last_pretrain_loss=training['history'][-1]['loss'],
        training_seconds=sum(r['seconds'] for r in training['history']))

repo=root/'work/tokenizer-pde-discovery'
family=set(repo.glob('src/**/*.py')) | set(repo.glob('eqgpt_original/code/*.py'))
payload={p.relative_to(repo).as_posix():digest(p) for p in family}
assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==run['code']
assert digest(repo/'scripts/pde_experiment.py')==run['runner']
provenance=json.loads((repo/'eqgpt_original/provenance.json').read_text())
for name,expected in provenance['files'].items():assert digest(repo/'eqgpt_original'/name)==expected
manifest=json.loads((repo/'data/formulas/636/manifest.json').read_text())
assert digest(repo/'data/formulas/636/manifest.json')==run['source_sha256']
for name,expected in manifest['files'].items():assert digest(repo/'data/formulas/636'/name)==expected
preflight=json.loads((root/'outputs/preflight636.json').read_text())
summary['repository_base_commit']=preflight['commit']
for case,report in preflight['mlp_caches'].items():
    assert digest(repo/'data/mlp_cache'/case/'surrogate_cache.pt')==report['cache_sha256']
summary['artifact_hashes_verified']=True
if (root/'outputs/storage_policy636.json').exists():
    summary['storage_policy']=json.loads((root/'outputs/storage_policy636.json').read_text())
if (root/'outputs/restarts636.json').exists():
    summary['recoveries']=json.loads((root/'outputs/restarts636.json').read_text())
    assert all(r.get('recovery_verified') for r in summary['recoveries'])
if (root/'outputs/storage_migration636.json').exists():
    summary['storage_migration']=json.loads((root/'outputs/storage_migration636.json').read_text())
(root/'outputs/summary636.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['636 条大语料实验已全部完成。','',
       '636 个源公式，20,352 条增强样本。仅运行 636 组；未运行 212 小语料实验。',
       '4 个模型各训练 100 epoch、15,900 个优化器更新；8 个案例各搜索 5×400 个候选，总计 64,000 个。',
       '预训练每 epoch 保存断点；搜索每 25 个候选、每轮保存断点。',
       '所有结果文件哈希、候选数量、微调轮数及共享数值缓存均已复核。','',
       '模型                    Top1正确案例  Top10含正确案例']
for arm,data in summary['models'].items():
    lines.append(f"{arm:23s} {data['top1_correct']}/8           {data['top10_correct']}/8")
lines += ['', '本仓库正式流程不含最终 PINN 系数优化；以上为 PDE 结构发现结果。',
          '完整候选、轮次轨迹、诊断及最终权重位于 large636/；源码快照位于 runtime-source.zip。',
          '为控制占用，诊断完成后释放第1至4轮中间权重；保留其哈希、预训练权重、最终权重和全部结果。']
(root/'outputs/summary636.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
