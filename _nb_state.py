import json, re
nb = json.load(open('examples/sslgraph/bench/compare_pretrain_quality.ipynb', 'r', encoding='utf-8'))
ansi = re.compile(r'\x1b\[[0-9;]*m')

# look at cells 13/14/15 outputs
for idx in (13, 14, 15):
    c = nb['cells'][idx]
    print(f'=== cell {idx} (id={c.get("id")}) ec={c.get("execution_count")} outs={len(c.get("outputs",[]))} ===')
    for i, o in enumerate(c.get('outputs', [])):
        ot = o.get('output_type')
        if ot == 'error':
            print(f'  [{i}] ERROR:', o.get('ename'), o.get('evalue'))
        elif ot == 'stream':
            txt = ''.join(o.get('text', []))
            if txt.strip():
                # only show non-tqdm interesting lines
                interesting = [ln for ln in txt.splitlines() if 'RMSE' in ln or '====' in ln or 'k_scaffold' in ln or 'cached' in ln or 'saved' in ln or 'side' in ln.lower()]
                if interesting:
                    print(f'  [{i}] [{o.get("name")}] ({len(txt.splitlines())} lines, filtered):')
                    for ln in interesting[:30]:
                        print('    ', ln)
                else:
                    print(f'  [{i}] [{o.get("name")}] {txt[-400:]!r}')
        elif ot == 'display_data' or ot == 'execute_result':
            txt = ''.join(o.get('data', {}).get('text/plain', []))
            if txt.strip():
                print(f'  [{i}] [{ot}] {txt[:1000]!r}')

# also dump finetune_summary.json files if present
from pathlib import Path
for d in sorted(Path('examples/sslgraph/bench/figs').glob('pretrain_quality_*')):
    s = d / 'finetune_summary.json'
    if s.is_file():
        print(f'\n--- {s} ---')
        print(s.read_text(encoding='utf-8'))
