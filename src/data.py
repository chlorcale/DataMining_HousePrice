"""Dataset IO, strict checks and leakage-safe feature extraction."""
import re
import pandas as pd
import numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/'data'/'ames.csv'

def canonical(s):
    s=re.sub(r'[^a-z0-9]+','_',str(s).lower()).strip('_')
    return s

def read_dataset(path=DATA):
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Copy the original 2930-row Ames CSV to data/ames.csv.")
    df=pd.read_csv(p,low_memory=False)
    df.columns=[canonical(x) for x in df.columns]
    if 'saleprice' in df: df=df.rename(columns={'saleprice':'sale_price'})
    if 'price' in df and 'sale_price' not in df: df=df.rename(columns={'price':'sale_price'})
    if 'sale_price' not in df:raise ValueError('CSV does not contain SalePrice / price target.')
    df['sale_price']=pd.to_numeric(df['sale_price'],errors='coerce')
    df=df[df.sale_price.gt(0) & df.sale_price.notna()].copy().reset_index(drop=True)
    if len(df) < 2000:raise ValueError(f'Expected full ~2930 rows of Ames housing, got {len(df)}; not silently using Kaggle 1460-row split.')
    # Explicitly exclude parcel identity and row identifiers, not predictors.
    X=df.drop(columns=[c for c in ['sale_price','pid','order','id','unnamed_0'] if c in df],errors='ignore').copy()
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
            s=X[c].astype('string').str.replace(',','',regex=False).str.strip()
            parsed=pd.to_numeric(s,errors='coerce')
            if parsed.notna().sum() > .94 * X[c].notna().sum() and X[c].notna().sum()>0:
                X[c]=parsed
            else:
                X[c]=X[c].astype('object')
    return X,df.sale_price.astype(float)

def summarize(X,y):
    return {'rows':int(len(y)),'predictors':int(X.shape[1]), 'target_min':float(y.min()),'target_median':float(y.median()),'target_max':float(y.max()),'target_missing':int(y.isna().sum()), 'missing_top':{k:round(float(v),5) for k,v in X.isna().mean().sort_values(ascending=False).head(15).items()},'unique_neighborhoods':int(X['neighborhood'].nunique()) if 'neighborhood' in X else None}
