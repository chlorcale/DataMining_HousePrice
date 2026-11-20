"""Shared CO3029 evaluation contract, NOT any member's algorithm implementation.

Only development data are available for CV/tuning. Test labels may be evaluated
once after a member freezes model choices. Split is deterministic, but the
holdout from the historical reference has been viewed previously: disclose this.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data import read_dataset
ROOT=Path(__file__).resolve().parents[1]
SEED=42
TEST_FRACTION=.20
FOLDS=5
EXCLUDED_FOR_PRE_SALE=('yr_sold','mo_sold','sale_type','sale_condition')

def load_split(pre_sale=True):
    X,y=read_dataset(ROOT/'data/ames.csv')
    if pre_sale:
        X=X.drop(columns=list(EXCLUDED_FOR_PRE_SALE),errors='ignore')
    ids=np.arange(len(y))
    train_ids,test_ids=train_test_split(ids,test_size=TEST_FRACTION,random_state=SEED,shuffle=True)
    return X.iloc[train_ids].copy(),X.iloc[test_ids].copy(),y.iloc[train_ids].copy(),y.iloc[test_ids].copy(), train_ids, test_ids

def folds():
    return KFold(n_splits=FOLDS,shuffle=True,random_state=2026)

def scores(y_true,pred):
    y=np.asarray(y_true,dtype=float); p=np.asarray(pred,dtype=float)
    if len(y)!=len(p) or not np.all(np.isfinite(p)):raise ValueError('Predictions must be finite and match ground truth length')
    if np.any(p<0):raise ValueError('Predicted sale prices must be nonnegative')
    return {'mae_usd':float(mean_absolute_error(y,p)),
            'rmse_usd':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p)),
            'mape_pct':float(np.mean(np.abs((y-p)/np.maximum(y,1)))*100)}

def validate_result(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    required={'owner','algorithm','pre_sale_features','seed','n_train','n_test','cv_mae_mean_usd','cv_mae_std_usd','test_metrics','notes'}
    if not required.issubset(obj):raise ValueError(f'Missing fields in {path}: {sorted(required-set(obj))}')
    if obj['seed']!=SEED:raise ValueError('Seed differs from team protocol')
    if obj['n_train']!=2344 or obj['n_test']!=586:raise ValueError('Different split, cannot compare directly')
    if not obj['pre_sale_features']:raise ValueError('Pre-sale valuation must exclude post-sale fields')
    expected={'mae_usd','rmse_usd','r2','mape_pct'}
    if not expected.issubset(obj['test_metrics']):raise ValueError('Missing evaluation metrics')
    for k in expected:
        if not np.isfinite(float(obj['test_metrics'][k])):raise ValueError('Nonfinite metrics')
    return obj
