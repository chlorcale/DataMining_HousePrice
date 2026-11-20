"""Export fully transformed numeric matrices for inspection/one-shot refit ONLY.

CRITICAL: These full-dev fitted CSVs CANNOT serve as CV input because they
would leak each validation fold's feature distributions into preprocessing.
Use src.preprocessing factories inside CV to get unbiased estimates.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from src.data import read_dataset
from src.protocol import holdout_indices
from src.preprocessing import make_linear_preprocessor,make_tree_preprocessor,make_lightgbm_preprocessor

def main():
    X,y=read_dataset();it,iv=holdout_indices(len(X));train=X.iloc[it];test=X.iloc[iv]
    out=ROOT/'data'/'fitted_previews';out.mkdir(parents=True,exist_ok=True)
    for name,factory in [('linear',make_linear_preprocessor),('tree',make_tree_preprocessor)]:
        processor=factory(); processor.fit(train)
        a=np.asarray(processor.transform(train));b=np.asarray(processor.transform(test))
        assert a.shape[1]==b.shape[1] and np.isfinite(a).all() and np.isfinite(b).all(),name
        colnames=[f'feature_{i:03d}' for i in range(a.shape[1])]
        pd.DataFrame(a,columns=colnames).to_csv(out/f'{name}_train_numeric.csv',index=False)
        pd.DataFrame(b,columns=colnames).to_csv(out/f'{name}_test_numeric.csv',index=False)
        names=processor.named_steps['pre'].get_feature_names_out().tolist()
        pd.DataFrame({'column':colnames,'transformer_feature':names}).to_csv(out/f'{name}_columns.csv',index=False)
        print(name,a.shape,b.shape)
    # LightGBM native categorical has pandas categorical dtype, not a numeric matrix.
    # Export a diagnostic numeric view with -1 for unknown category only. Missing
    # numeric values remain NaN intentionally (native LightGBM manages NaNs).
    processor=make_lightgbm_preprocessor();processor.fit(train)
    train_frame=processor.transform(train);test_frame=processor.transform(test)
    categorical=[c for c in train_frame if isinstance(train_frame[c].dtype,pd.CategoricalDtype)]
    numeric_codes=[]
    for frame in (train_frame,test_frame):
        t=frame.copy()
        for c in categorical:t[c]=t[c].cat.codes.astype('int32')
        numeric_codes.append(t)
    numeric_codes[0].to_csv(out/'lightgbm_train_numeric_view.csv',index=False)
    numeric_codes[1].to_csv(out/'lightgbm_test_numeric_view.csv',index=False)
    (out/'lightgbm_category_levels.json').write_text(json.dumps({c:list(train_frame[c].cat.categories) for c in categorical},indent=2,ensure_ascii=False),encoding='utf8')
    (out/'README.md').write_text(
        'CAUTION: These files are fitted using the full 2344-row development split. '
        'For cross-validation, do NOT train on these CSVs: fit the preprocessing '
        'factory separately within each training fold. The LightGBM numeric view '
        'is diagnostic only and uses category codes (-1 unseen); use a fitted '
        'make_lightgbm_preprocessor() to preserve native pandas categories. '
        'For the final model after selection, fitting on all development data is allowed.\n',
        encoding='utf8')
    print('WARNING: Previews are NOT valid inputs to CV; see README in fitted_previews/')
if __name__=='__main__':main()
