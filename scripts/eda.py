"""Group-owned EDA (development partition only) with actual saved plots and tables."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd,numpy as np
from src.data import read_dataset
from src.protocol import holdout_indices

def main():
    X,y=read_dataset();dev,_=holdout_indices(len(y));a=X.iloc[dev];p=y.iloc[dev]
    out=ROOT/'report'/'figures';out.mkdir(parents=True,exist_ok=True)
    stats=ROOT/'data'/'eda';stats.mkdir(parents=True,exist_ok=True)
    p.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_csv(stats/'dev_price_summary.csv')
    a.isna().mean().sort_values(ascending=False).rename('missing_fraction').to_csv(stats/'dev_missing_fraction.csv')
    a.dtypes.astype(str).rename('dtype').to_csv(stats/'dev_feature_types.csv')
    if 'overall_qual' in a:
        pd.DataFrame({'quality':a.overall_qual,'price':p}).groupby('quality').price.agg(['count','median','mean']).to_csv(stats/'dev_price_by_quality.csv')
    if 'neighborhood' in a:
        pd.DataFrame({'neighborhood':a.neighborhood,'price':p}).groupby('neighborhood').price.agg(['count','median','mean']).sort_values('count',ascending=False).to_csv(stats/'dev_price_by_neighborhood.csv')
    numeric=a.select_dtypes(include=np.number).copy();numeric['sale_price']=p
    numeric.corr(numeric_only=True).sale_price.drop('sale_price').sort_values(key=abs,ascending=False).rename('correlation').to_csv(stats/'dev_numeric_price_correlation.csv')
    area=pd.to_numeric(a['gr_liv_area'],errors='coerce')
    unit=p/area.replace(0,np.nan)
    logunit=np.log(unit.replace([np.inf,-np.inf],np.nan)).dropna()
    q1,q3=logunit.quantile([.25,.75]);low=q1-1.5*(q3-q1);high=q3+1.5*(q3-q1)
    flags=~np.log(unit).between(low,high); flags=flags & np.log(unit).notna()
    pd.DataFrame({'raw_row_id':dev,'sale_price':p.to_numpy(),'gr_liv_area':area.to_numpy(),'price_per_sqft':unit.to_numpy(),'flag_review_only':flags.to_numpy()}).to_csv(stats/'dev_outlier_review.csv',index=False)
    (stats/'outlier_policy.json').write_text(json.dumps({'log_unit_price_iqr_lower':float(low),'log_unit_price_iqr_upper':float(high),'review_flagged':int(flags.sum()),'policy':'Do NOT remove from shared dataset; no target-derived filtering of holdout/CV validation.'},indent=2),encoding='utf8')
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(p,bins=40);ax.set(xlabel='Sale price (USD)',ylabel='Number of homes',title='Ames development price distribution');fig.tight_layout();fig.savefig(out/'price_distribution.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter(a['gr_liv_area'],p,s=9,alpha=.35);ax.set(xlabel='Above-ground living area (sq ft)',ylabel='Sale price (USD)',title='Living area vs price, development only');fig.tight_layout();fig.savefig(out/'area_price.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.barh(a.isna().mean().sort_values(ascending=False).head(12).index[::-1],a.isna().mean().sort_values(ascending=False).head(12).values[::-1]);ax.set(xlabel='Missing fraction',title='Top missing predictors');fig.tight_layout();fig.savefig(out/'missing_fraction.png',dpi=150);plt.close(fig)
    print('EDA saved',len(p),'development rows, outlier review flags:',flags.sum())
if __name__=='__main__':main()
