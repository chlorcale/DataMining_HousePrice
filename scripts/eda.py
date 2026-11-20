"""Descriptive analysis on development partition only; no model selection."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.protocol import load_split
from src.data import summarize
X,_,y,_,_,_=load_split(pre_sale=True)
out=ROOT/'outputs'/'eda';out.mkdir(parents=True,exist_ok=True)
figdir=ROOT/'report'/'figures';figdir.mkdir(parents=True,exist_ok=True)
summary=summarize(X,y)
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
y.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_csv(out/'price_describe.csv')
X.isna().mean().sort_values(ascending=False).rename('missing_fraction').to_csv(out/'missing.csv')
for cat in ('neighborhood','overall_qual'):
    pd.DataFrame({'group':X[cat],'sale_price':y}).groupby('group',dropna=False).sale_price.agg(['count','median','mean']).to_csv(out/f'price_by_{cat}.csv')
num=X.select_dtypes(include=np.number).copy()
num['sale_price']=y.to_numpy()
num.corr(numeric_only=True)['sale_price'].drop('sale_price').sort_values(key=lambda v:v.abs(),ascending=False).to_csv(out/'numeric_correlations.csv')
fig,ax=plt.subplots(figsize=(8,4));ax.hist(y,bins=42);ax.set(xlabel='Sale price (USD)',ylabel='Count',title='Ames sale-price distribution — development data');fig.tight_layout();fig.savefig(out/'price_hist.png',dpi=160);fig.savefig(figdir/'price_hist.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));ax.scatter(X['gr_liv_area'],y,s=8,alpha=.35);ax.set(xlabel='Above-ground living area (ft²)',ylabel='Sale price (USD)',title='Price vs living area — development data');fig.tight_layout();fig.savefig(out/'price_vs_area.png',dpi=160);fig.savefig(figdir/'price_vs_area.png',dpi=160);plt.close(fig)
q=X['overall_qual'];groups=pd.DataFrame({'quality':q,'price':y}).groupby('quality').price.median()
fig,ax=plt.subplots(figsize=(8,4));ax.bar(groups.index.astype(str),groups.values);ax.set(xlabel='Overall quality',ylabel='Median price (USD)',title='Price by quality — development data');fig.tight_layout();fig.savefig(out/'quality.png',dpi=160);fig.savefig(figdir/'quality.png',dpi=160);plt.close(fig)
# Audit only: IQR flags on development log-price-per-ft², never a deletion rule.
area=pd.to_numeric(X['gr_liv_area'],errors='coerce')
ratio=y/area.replace(0,np.nan)
logratio=np.log(ratio.replace([np.inf,-np.inf],np.nan)).dropna()
q1,q3=logratio.quantile([.25,.75]);spread=q3-q1
flags=(np.log(ratio)<q1-1.5*spread)|(np.log(ratio)>q3+1.5*spread)
flags=flags.fillna(False)
pd.DataFrame({'sale_price':y,'gr_liv_area':area,'price_per_ft2':ratio,'flag_iqr':flags}).sort_values('price_per_ft2',ascending=False).head(30).to_csv(out/'outlier_review.csv',index=False)
(out/'outlier_audit.json').write_text(json.dumps({'train_development_rows':int(len(y)),'flagged_count':int(flags.sum()),'flagged_fraction':float(flags.mean()),'rule':'IQR on log(SalePrice / GrLivArea), dev-only exploratory; do not auto-delete'},indent=2),encoding='utf8')
print('EDA complete:',out)

