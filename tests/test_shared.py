import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from src.data import read_dataset
from src.features import FeatureBuilder
from src.protocol import holdout_indices,make_cv
from src.preprocessing import make_linear_preprocessor,make_tree_preprocessor,make_lightgbm_preprocessor
ROOT=Path(__file__).resolve().parents[1]

def test_exact_ultra_hashes():
    m=json.loads((ROOT/'data/processed/manifest.json').read_text())
    for rel,key in [('data/ames.csv','source_data_sha256'),('src/data.py','source_data_py_sha256'),('src/features.py','source_features_py_sha256')]:
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==m[key]

def test_holdout_and_fe_rows():
    X,y=read_dataset();tr,te=holdout_indices(len(y))
    assert len(tr)==2344 and len(te)==586 and len(set(tr)&set(te))==0
    assert X.shape==(2930,79)
    assert FeatureBuilder().fit_transform(X).shape[1]==86

def test_all_three_preprocessors_run_and_keep_train_schema():
    X,_=read_dataset();tr,te=holdout_indices(len(X));a=X.iloc[tr[:240]];b=X.iloc[te[:32]]
    for f in (make_linear_preprocessor,make_tree_preprocessor):
        p=f().fit(a);aa=np.asarray(p.transform(a));bb=np.asarray(p.transform(b))
        assert aa.shape[1]==bb.shape[1]
        assert np.isfinite(aa).all() and np.isfinite(bb).all()
    lg=make_lightgbm_preprocessor().fit(a)
    q=lg.transform(b)
    assert q.shape[1]==86
    assert any(isinstance(q[c].dtype,pd.CategoricalDtype) for c in q)

def test_unknown_categorical_never_fit_on_test():
    X,_=read_dataset();tr,te=holdout_indices(len(X));train=X.iloc[tr[:100]];val=X.iloc[te[:3]].copy()
    val['neighborhood']='UNSEEN_NEIGHBORHOOD_NEVER_TRAIN'
    pre=make_lightgbm_preprocessor().fit(train);frame=pre.transform(val)
    assert frame['neighborhood'].isna().all()

def test_encoded_train_previews_have_no_missing():
    for fam in ['linear','tree']:
        a=pd.read_csv(ROOT/f'data/fitted_previews/{fam}_train_numeric.csv')
        b=pd.read_csv(ROOT/f'data/fitted_previews/{fam}_test_numeric.csv')
        assert len(a)==2344 and len(b)==586 and list(a.columns)==list(b.columns)
        assert np.isfinite(a.to_numpy()).all() and np.isfinite(b.to_numpy()).all()

def test_fold_safe_cv_smoke():
    X,y=read_dataset();it,_=holdout_indices(len(X));a=X.iloc[it[:100]];b=y.iloc[it[:100]]
    from sklearn.compose import TransformedTargetRegressor
    model=TransformedTargetRegressor(regressor=Pipeline([('preprocess',make_linear_preprocessor()),('estimator',Ridge(alpha=25))]),func=np.log1p,inverse_func=np.expm1,check_inverse=False)
    scores=cross_val_score(model,a,b,cv=3,scoring='neg_mean_absolute_error')
    assert np.isfinite(scores).all()
