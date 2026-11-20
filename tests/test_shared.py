from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pytest,numpy as np
from src.protocol import load_split,folds,scores
from src.data import read_dataset
from src.features import FeatureBuilder

def test_dataset_and_split():
 X,y=read_dataset();assert len(y)==2930 and X.shape[1]==79
 a,b,ya,yb,ia,ib=load_split();assert len(ya)==2344 and len(yb)==586
 assert not set(ia).intersection(ib)
 for x in ('yr_sold','mo_sold','sale_type','sale_condition'):assert x not in a
 assert 'sale_price' not in a and 'pid' not in a

def test_engineered_ames_names():
 X,_=read_dataset(); z=FeatureBuilder().transform(X.iloc[:20]);assert 'living_plus_total_basement_sf' in z
 assert np.allclose(z['living_plus_total_basement_sf'],X.iloc[:20]['gr_liv_area']+X.iloc[:20]['total_bsmt_sf'].fillna(0),equal_nan=True)
 assert np.allclose(z['porch_total'], X.iloc[:20][['open_porch_sf','enclosed_porch','screen_porch','3ssn_porch']].fillna(0).sum(axis=1),equal_nan=True)

def test_metrics():
 x=scores([100,200],[110,190]);assert x['mae_usd']==10 and x['r2']>0

def test_folds_train_only():
 a,_,_,_,_,_=load_split(); foldlist=list(folds().split(a));assert len(foldlist)==5
 assert all(set(t).isdisjoint(v) for t,v in foldlist)

def test_no_post_sale_leakage_in_engineered():
 a,_,_,_,_,_=load_split()
 result=FeatureBuilder().transform(a.iloc[:5])
 assert 'age_at_sale' not in result and 'years_since_remodel' not in result
 assert not {'yr_sold','mo_sold','sale_type','sale_condition'}.intersection(result.columns)
