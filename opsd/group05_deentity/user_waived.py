"""User-authorized cancellation of remaining sampling; never invent reviews."""
import json
from pathlib import Path
from copy import deepcopy
from opsd.common import digest
from opsd.dev_eval.prepare import ROOT, sha, jsonl
from . import api_extract as api
from . import api_sampling as sampling
from .entities import candidate, validate_candidate
from .audit import reviewed

PROVIDER = 'external_api_user_waived'
CONFIG = Path('opsd/group05_deentity/config.training.toml')
AUTH = {'date':'2026-10-27', 'instruction':'不用抽查了，直接启动服务器进行训练，实验方案你还记得吗',
        'scope':'remaining risk/sample audit waived; program/source/order checks retained',
        'human_semantic_review_claimed':False}

def check_provenance(manifest, selected, root=ROOT):
    if (manifest.get('provider') != PROVIDER or manifest.get('sample_audit_complete') is not False
        or manifest.get('semantic_review_complete') is not False or manifest.get('audit_waiver') != AUTH):
        raise ValueError('Missing explicit waiver or false audit claim')
    run_path = api.safe_path(manifest['run_manifest_path'], root)
    if sha(run_path) != manifest['run_manifest_sha256']:
        raise ValueError('Extraction source manifest changed')
    run = json.loads(run_path.read_text(encoding='utf-8')); spec=run['spec']
    api.verify_files(spec['frozen_files_sha256'], root)
    api.verify_files(manifest['artifacts_sha256'], root)
    api.verify_files(manifest['code_sha256'], root)
    if digest(run['imported_agent_records']) != spec['imported_agent_sha256']:
        raise ValueError('Imported annotation evidence changed')
    if (manifest['order_ids'] != [r['id'] for r in selected] or spec['order_ids'] != manifest['order_ids']
        or manifest['train_sha256'] != spec['train_sha256'] or manifest['api'] != spec['api']):
        raise ValueError('Source/order/model mismatch')
    records=api.read_records(run_path.parent, selected, digest(spec))
    imported=run['imported_agent_records']; fall={}
    for i,row in enumerate(selected):
        if row['id'] in imported: continue
        r=records[row['id']]; path=run_path.parent/'records'/f'{i+1:04d}.json'
        if r['status'] != 'validated' or manifest['artifacts_sha256'].get(api.rel(path,root)) != sha(path):
            raise ValueError('Unvalidated or unbound API record')
        if r.get('processing_mode')=='original_fallback':
            if r['candidate']['rubrics'] != row['rubrics'] or r['candidate']['changes']:
                raise ValueError('Fallback differs from original')
            fall[row['id']]=r['fallback_reason']
    if fall != manifest['original_fallback_reasons']:
        raise ValueError('Fallback accounting differs')

def main():
    c,_,rows,_,_=api.agent.sources(CONFIG)
    base=ROOT/Path(c['derived_data']).parent/'api'/c['api_run_name']
    if (base/'active.lock').exists(): raise ValueError('API workers active')
    targets=[ROOT/c[k] for k in ('derived_data','review_data','extraction_manifest')]
    if any(p.exists() for p in targets): raise FileExistsError('Final outputs already exist')
    run_path=base/'run.json'; run=json.loads(run_path.read_text(encoding='utf-8')); spec=run['spec']
    api.verify_files(spec['frozen_files_sha256'])
    records=api.read_records(base,rows,digest(spec)); imported=run['imported_agent_records']
    if any(records.get(r['id'],{}).get('status')!='validated' for r in rows if r['id'] not in imported):
        raise ValueError('Unresolved program errors')
    checked=sampling.reviews(base,rows,records); required=sampling.required_ids(rows,records,imported)
    artifacts={api.rel(run_path):sha(run_path),**spec['frozen_files_sha256']}
    for i,row in enumerate(rows):
        if row['id'] not in imported:
            path=base/'records'/f'{i+1:04d}.json'; artifacts[api.rel(path)]=sha(path)
    for _,path in checked.values(): artifacts[api.rel(path)]=sha(path)
    fall={uid:r['fallback_reason'] for uid,r in records.items() if r.get('processing_mode')=='original_fallback'}
    manifest={'provider':PROVIDER,'rule_version':api.RULE_VERSION,'records':len(rows),
        'order_ids':[r['id'] for r in rows], 'train_sha256':spec['train_sha256'],
        'baseline_actual_prefix_sha256':spec['baseline_actual_prefix_sha256'], 'api':spec['api'],
        'run_manifest_path':api.rel(run_path),'run_manifest_sha256':sha(run_path),
        'artifacts_sha256':artifacts,'code_sha256':{api.rel(Path(__file__).resolve()):sha(Path(__file__))},
        'derived_data_path':api.rel(targets[0]),'review_data_path':api.rel(targets[1]),
        'audit_waiver':AUTH,'sample_audit_complete':False,'semantic_review_complete':False,
        'human_review_claimed':False,'remaining_audit_ids_waived':[uid for uid in required if uid not in checked],
        'original_fallback_records':len(fall),'original_fallback_reasons':fall,
        'source_counts':{'agent_annotations':len(imported),'external_api':len(rows)-len(imported)},
        'actual_agent_review_ids':list(checked),'fallback_is_not_deentity_success':True}
    items=[]; reviews=[]; fingerprint=digest(manifest)
    for row in rows:
        uid=row['id']; evidence=imported.get(uid) or records[uid]
        item=candidate(row,evidence['entities'],fingerprint)
        item['annotation_provider']=evidence['origin']
        if uid in fall:
            item.update(processing_mode='original_fallback',deentity_success=False,fallback_reason=fall[uid],
                        annotation_provider='user_authorized_original_fallback')
        review=deepcopy(checked[uid][0] if uid in checked else evidence['review'])
        validate_candidate(row,item); items.append(item); reviews.append(review)
    reviewed(rows,items,reviews); check_provenance(manifest,rows)
    for path,values in zip(targets[:2],(items,reviews)):
        with path.open('x',encoding='utf-8') as f:
            for value in values: f.write(json.dumps(value,ensure_ascii=False)+'\n')
    api.agent.fresh_json(targets[2],manifest)
    report=api.REPORTS/'API_DATA_PREPARATION.md'; report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(f'# 第五组数据准备\n\n共{len(rows)}条，原agent复用{len(imported)}条，其余1020条API处理。\n\n151条异常整份原Rubric回退（7.55%），不计作去实体成功。用户明确取消剩余{len(manifest["remaining_audit_ids_waived"])}条风险/固定抽查；sample_audit_complete=false，未做全量语义审核，未伪造审核记录。\n\n程序重建、来源、顺序及哈希检查均保留。训练目标、超参和输入接口不变。\n',encoding='utf-8')
    print(json.dumps({'finalized':True,'records':len(rows),'original_fallback':len(fall),
                      'remaining_audit_waived':len(manifest['remaining_audit_ids_waived'])}))

if __name__=='__main__': main()
