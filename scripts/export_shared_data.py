"""Create faithful shared FeatureBuilder output, with frozen row IDs and manifest."""
import sys,hashlib,json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.data import read_dataset
from src.features import FeatureBuilder
from src.protocol import holdout_indices,make_cv,SEED,TEST_SIZE,CV_SEED

def main():
    X,y=read_dataset()
    it,iv=holdout_indices(len(y))
    out=ROOT/'data'/'processed';out.mkdir(parents=True,exist_ok=True)
    train=FeatureBuilder(engineered=True).fit_transform(X.iloc[it].copy())
    test=FeatureBuilder(engineered=True).fit_transform(X.iloc[iv].copy())
    train.reset_index(drop=True).to_csv(out/'train_features.csv',index=False)
    test.reset_index(drop=True).to_csv(out/'test_features.csv',index=False)
    pd.DataFrame({'sale_price':y.iloc[it].to_numpy()}).to_csv(out/'train_targets.csv',index=False)
    pd.DataFrame({'sale_price':y.iloc[iv].to_numpy()}).to_csv(out/'test_targets.csv',index=False)
    pd.DataFrame({'raw_row_id':it}).to_csv(out/'train_row_ids.csv',index=False)
    pd.DataFrame({'raw_row_id':iv}).to_csv(out/'test_row_ids.csv',index=False)
    folds=[]
    for i,(tr,va) in enumerate(make_cv().split(train),1):
        for idx in tr:folds.append({'fold':i,'dev_row':int(idx),'role':'train'})
        for idx in va:folds.append({'fold':i,'dev_row':int(idx),'role':'validation'})
    pd.DataFrame(folds).to_csv(out/'cv_folds.csv',index=False)
    file_hash=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    meta={
        'source_archive':'BTL2(1).zip',
        'source_data_sha256':file_hash(ROOT/'data'/'ames.csv'),
        'source_data_py_sha256':file_hash(ROOT/'src'/'data.py'),
        'source_features_py_sha256':file_hash(ROOT/'src'/'features.py'),
        'raw_samples':len(X),'raw_predictors':X.shape[1],
        'features_after_ultra_feature_builder':train.shape[1],
        'train_rows':len(train),'test_rows':len(test),
        'split':'sklearn train_test_split, test_size=0.2, random_state=42, shuffle=True',
        'cv':'KFold(5,shuffle=True,random_state=123); 2026 in optional Ultra advanced analysis',
        'source_feature_builder':'FeatureBuilder(engineered=True) copied byte-for-byte',
        'processing_level':'rowwise FE only; fold-fitted imputation/encoding/scaling in src/preprocessing.py',
        'test_target_policy':'For final once-only evaluation; do not use for feature selection or tuning',
        'categorical_missing_policy':'retained at shared FE stage; LightGBM native pathway keeps NaN',
    }
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:meta[k] for k in ['raw_samples','raw_predictors','features_after_ultra_feature_builder','train_rows','test_rows']},indent=2))
if __name__=='__main__':main()
