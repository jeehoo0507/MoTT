"""Four matched ablations, three real RL phases, and auditable learning histories."""
import csv
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts import curriculum_campaign as campaign
from scripts import pkg_guard_study as study
from decap_loop.post_training import ARMS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/posttrain_v1'
BRANCH='report-models-mott-posttrain-v1'


def learning_report():
    rows=[];previous={}
    for path in sorted((OUT/'models').glob('*/phase*/trace.jsonl')):
        for line in path.read_text().splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if event['kind']!='validation':continue
            for group,records in event['rows'].items():
                solved=[r for r in records if r['success']]
                key=(path.parent.parent.name,group);old=previous.get(key,{})
                common=[(old[r['instance']],r) for r in records if r['instance'] in old]
                joint=[(x,y) for x,y in common if x['success'] and y['success']]
                rows.append(dict(model=path.parent.parent.name,phase=event['phase'],stage=event['stage'],iteration=event['iteration'],
                    phase_seconds=event['elapsed'],group=group,n=len(records),success_rate=len(solved)/len(records),
                    count_success=sum(r['count'] for r in solved)/len(solved) if solved else None,
                    mean_positive_gap=sum(max(0,r['gap']) for r in records)/len(records),
                    removals=sum(r['moves']['remove'] for r in records),replacements=sum(r['moves']['replace'] for r in records),
                    paired_n=len(common),joint_n=len(joint),
                    paired_success_change=sum(int(y['success'])-int(x['success']) for x,y in common)/len(common) if common else None,
                    joint_count_change=sum(y['count']-x['count'] for x,y in joint)/len(joint) if joint else None,
                    checkpoint=str(path.parent.relative_to(OUT)/event['checkpoint'])))
                previous[key]={r['instance']:r for r in records}
    if rows:
        with (OUT/'learning_curves.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Three-phase post-training: validation learning history','',
        'Exploratory seed-0 results first; validation is not independent test performance.',
        'Checkpoints use iterations and elapsed time, not epochs. Only final arm comparisons have matched 45-minute budgets.',
        'No causal conclusion is automated: compare the ablations and jointly solved boards.',
        'Stage-3 archive success does not guarantee empty-start policy success; validation always starts empty.','',
        '| Model | Phase/stage | Iteration | Group | Success | Count on successes | Positive exact/grid gap |',
        '|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        c='NA' if r['count_success'] is None else f'{r["count_success"]:.2f}'
        lines.append(f'| {r["model"]} | {r["phase"]}/{r["stage"]} | {r["iteration"]} | {r["group"]} | {r["success_rate"]:.3f} | {c} | {r["mean_positive_gap"]:.5f} |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def train_arm(seed,arm,folder,cfg,src,model_args):
    previous=src;chain=[study.sha(src)];phase_metrics=[]
    for phase in (1,2,3):
        target=folder/f'phase{phase}';state=folder/f'phase{phase-1}/post_state.json'
        if not (target/'phase_complete.json').exists():
            cmd=[sys.executable,'scripts/train_post_training.py','--post_arm',arm,'--post_phase',str(phase),
                '--root_init',str(src),'--certify_workers',str(cfg['workers']),'--stage','rl','--model','mott','--peak_refine',
                '--init',str(previous),'--out',str(target),'--pool',str(OUT/'inputs/pool'),
                '--teacher_dir',str(OUT/'inputs/teacher'),'--seed',str(seed),
                '--hours',str(cfg['train_hours']/3),'--iters','1000000','--B','4','--M','8','--chunk','1',
                '--lr','0.0001','--lam','20','--imit','0','--certify','2','--val_every','25','--ckpt_min','5','--max_steps','64']
            if phase>1:cmd+=['--post_state',str(state)]
            for key in ('d','layers','heads'):cmd+=['--'+key,str(model_args[key])]
            for flag in ('fc_bias','mode_head'):
                if model_args.get(flag):cmd+=['--'+flag]
            campaign.owned(cmd,f'{arm}_s{seed}_phase{phase}',gpu=True)
        ck=torch.load(target/'best.pt',map_location='cpu',weights_only=False)
        if study.sha(ck['args']['init'])!=study.sha(previous):raise ValueError('Broken phase checkpoint chain')
        if study.sha(ck['args']['root_init'])!=study.sha(src):raise ValueError('Wrong root initialization')
        previous=target/'best.pt';chain.append(study.sha(previous))
        metrics=json.loads((target/'metrics.json').read_text())
        phase_metrics.append({k:metrics[k] for k in ('iters','hours','best_iter','cuda_oom_skipped','best_val_score')})
        learning_report()
    # Never export a model as complete until all three budget-matched phases finished.
    shutil.copy2(previous,folder/'best.pt')
    shutil.copy2(folder/'phase3/validation_best.json',folder/'validation_best.json')
    study.save(folder/'metrics.json',dict(iters=sum(m['iters'] for m in phase_metrics),hours=sum(m['hours'] for m in phase_metrics),
        best_iter=phase_metrics[-1]['best_iter'],cuda_oom_skipped=sum(m.get('cuda_oom_skipped',0) for m in phase_metrics),
        phase_metrics=phase_metrics,chain_sha256=chain))


def gate():
    def compare(left,right):
        paths=[OUT/f'models/{arm}_s0/validation_best.json' for arm in (left,right)]
        return campaign.expansion_gate(*(json.loads(p.read_text()) for p in paths))
    against_control=compare('pt_control','pt_full');against_stage12=compare('pt_robust','pt_full')
    state=json.loads((OUT/'models/pt_full_s0/phase3/post_state.json').read_text())
    active=state['stats']['refinement_episodes']>0
    decision=dict(expand=against_control['expand'] and against_stage12['expand'] and active,against_control=against_control,against_stage12=against_stage12,refinement_active=active)
    study.save(OUT/'seed0_complete.json',dict(completed=time.time(),decision=decision))
    learning_report();return decision


def configure():
    campaign.OUT=OUT;campaign.BRANCH=BRANCH;campaign.ARMS=ARMS;campaign.PROTOCOL='POSTTRAIN_PROTOCOL.md'
    campaign.TRAIN_ARM_HOOK=train_arm;campaign.gate_from_models=gate


if __name__=='__main__':
    import signal
    def stop(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);configure()
    try:campaign.run()
    except BaseException as e:
        study.save(OUT/'status.json',dict(stage='failed_or_interrupted',error=str(e),updated=time.time()));raise
