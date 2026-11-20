"""Report real member results only: no synthetic baseline metrics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
items={'member_a':['ridge','random_forest'],'member_b':['elasticnet','lightgbm']}
for person,models in items.items():
    for model in models:
        f=ROOT/'results'/person/f'{model}_metrics.json'
        if f.exists():
            data=json.loads(f.read_text(encoding='utf8'))
            print('DONE',person,model,{k:data.get(k) for k in ['cv_mae','test_mae','test_rmse','test_r2','test_mape_pct']})
        else:print('PENDING',person,model,'expected:',f.relative_to(ROOT))
