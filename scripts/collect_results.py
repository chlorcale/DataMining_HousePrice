"""Integrate final reports without inventing missing results."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.protocol import validate_result
entries=[]
for owner in ('member_a','member_b'):
    files=sorted((ROOT/'results'/owner).glob('*.json'))
    if len(files)!=2:print(f'PENDING: {owner} requires exactly 2 algorithm JSON files; found {len(files)}')
    for f in files:
        x=validate_result(f);x['file']=str(f.relative_to(ROOT));entries.append(x)
seen=[x['algorithm'] for x in entries]
if len(set(seen))!=len(seen):raise SystemExit('Duplicate algorithm keys')
out=ROOT/'outputs'/'integration';out.mkdir(parents=True,exist_ok=True)
cols=['owner','algorithm','cv_mae_mean_usd','cv_mae_std_usd','test_mae_usd','test_rmse_usd','test_r2','test_mape_pct','file']
with (out/'model_comparison.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
    for r in entries:
        m=r['test_metrics'];w.writerow({'owner':r['owner'],'algorithm':r['algorithm'],'cv_mae_mean_usd':r['cv_mae_mean_usd'],'cv_mae_std_usd':r['cv_mae_std_usd'],'test_mae_usd':m['mae_usd'],'test_rmse_usd':m['rmse_usd'],'test_r2':m['r2'],'test_mape_pct':m['mape_pct'],'file':r['file']})
print(f'Imported {len(entries)}/4 algorithm summaries. File: {out/"model_comparison.csv"}')
if len(entries)!=4:print('REPORT REMAINS IN DRAFT MODE until both members deliver')
