"""Export completed Delta/GRU pilot evidence for the report; no training/evaluation started."""
import argparse
import csv
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/delta_memory_v1');p.add_argument('--out',default='reports/pkg_base_delta')
    a=p.parse_args();run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source=run/'paired_comparisons.json'
    paired=json.loads(source.read_text()) if source.exists() else {'contrasts':[]}
    records=[]
    for path in sorted((run/'eval').glob('*.json')):
        d=json.loads(path.read_text());condition=d['pool'].split('_')[1]
        tech=''.join(c for c in condition if c.isalpha());size=int(''.join(c for c in condition if c.isdigit()))
        for row in d['rows']:
            records.append(dict(tech=tech,size=size,arm=d['arm'],seed=d['seed'],pool=d['pool'],
                instance=row['instance'],success=row['success'],count=row['count'],worst_log10=row['worst'],
                placement_seconds=row['placement_seconds'],policy_seconds=row.get('policy_seconds'),
                certification_seconds=row.get('certification_seconds'),physics_mse=row.get('physics_mse')))
    if not records:raise SystemExit('No evaluation JSON found; no report generated')
    with (out/'all_cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    package=[c for c in paired['contrasts'] if c['tech']=='pkg']
    (out/'package_paired.json').write_text(json.dumps(package,indent=2)+'\n')
    lines=['# Delta/GRU 보고서용 탐색 결과','',
        '별도 짧은 학습 실험. 같은 문제의 시드 반복을 독립 문제로 세지 않는다. 본평가 수치에 합치지 않는다.',
        '시간은 CPU 평가 동시 실행의 영향을 받을 수 있어 전용 반복 측정 전에는 확정적 속도 향상을 주장하지 않는다.',
        '', '| 종류 | 크기 | 모델 | 시드 수 | 성공/평가 시도 | 평균 배치 초 | 성공 시 평균 부품 수 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for key in sorted({(r['tech'],r['size'],r['arm']) for r in records}):
        rows=[r for r in records if (r['tech'],r['size'],r['arm'])==key]
        successes=[r for r in rows if r['success']]
        cnt=f"{sum(r['count'] for r in successes)/len(successes):.2f}" if successes else '해당 없음'
        mean_time=sum(r['placement_seconds'] for r in rows)/len(rows)
        lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {len({r["seed"] for r in rows})} | {len(successes)}/{len(rows)} | {mean_time:.4f} | {cnt} |')
    lines+=['','부품 수 열은 성공한 문제의 구성이 달라 직접 우열을 판단하면 안 된다. 아래 paired 결과의 공동 성공 비교를 사용한다.','',
            '## 패키지형 paired 비교','']
    for c in package:
        lines.append(f'### {c["size"]}×{c["size"]}: {c["treated"]} − {c["base"]}')
        lines.append('')
        for s in c['per_seed']:
            lines.append(f'- seed {s["seed"]}: 성공률 차 {s["success_difference"]*100:.1f}%p, 구제 {s["rescued"]}, 새 실패 {s["harmed"]}, 공동 성공 {s["joint_success_n"]}, 공동 성공 부품 수 차 {s["count_difference"]}')
        lines.append('')
    lines+=['## 해석 범위','',
        'Delta+보조 학습의 pkg10 개선과 보조 학습 없는 Delta의 pkg16 개선은 서로 다른 후보의 탐색 신호다.',
        '같은 최종 구조가 모든 크기에서 개선됐다고 해석하지 않는다. GRU 대조군과 물리 보조 학습 대조군도 함께 보고한다.',
        '정확 행동 효과/마스크는 유지되므로 물리 공식을 학습하여 시뮬레이터를 대체했다는 결론은 낼 수 없다.','']
    (out/'ASSESSMENT.md').write_text('\n'.join(lines))
    meta=[]
    for path in sorted((run/'models').glob('*/complete.json')):
        d=json.loads(path.read_text());meta.append(dict(arm=d['arm'],seed=d['seed'],parameters=d['parameters'],
            train_seconds=d['train_seconds'],best=d['best'],bc=d['bc'],rl=d['rl']))
    (out/'training_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    archive=out.with_name(out.name+'_evidence.tgz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in out.glob('*'):
            if path.is_file():tar.add(path,arcname='report/'+path.name)
        for pattern in ('config.json','data_manifest.json','summary.csv','paired_comparisons.json','eval/*.json','models/*/complete.json'):
            for path in run.glob(pattern):tar.add(path,arcname='run/'+str(path.relative_to(run)))
    print('Report:',out/'ASSESSMENT.md');print('Evidence archive (no checkpoints):',archive)


if __name__=='__main__':main()
