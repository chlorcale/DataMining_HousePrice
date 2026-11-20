from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def col(df,name):
    if name in df:return pd.to_numeric(df[name],errors='coerce')
    return pd.Series(np.nan,index=df.index)
class FeatureBuilder(BaseEstimator,TransformerMixin):
    def __init__(self,engineered=True):self.engineered=engineered
    def fit(self,X,y=None):return self
    def transform(self,X):
        X=X.copy()
        if not self.engineered:return X
        q=col(X,'overall_qual'); a=col(X,'gr_liv_area')
        if a.isna().all():a=col(X,'area')
        X['quality_x_living_area']=q*a
        sold=col(X,'yr_sold');year=col(X,'year_built')
        X['age_at_sale']=(sold-year).clip(lower=0)
        X['years_since_remodel']=(sold-col(X,'year_remod_add')).clip(lower=0)
        X['total_bathrooms']=col(X,'full_bath')+0.5*col(X,'half_bath')+col(X,'bsmt_full_bath')+0.5*col(X,'bsmt_half_bath')
        X['porch_total']=sum(col(X,c).fillna(0) for c in ['open_porch_sf','enclosed_porch','screen_porch','3ssn_porch'])
        # Includes unfinished basement: this is NOT finished floor area.
        X['living_plus_total_basement_sf']=a+col(X,'total_bsmt_sf').fillna(0)
        X['garage_presence']=col(X,'garage_cars').gt(0).astype(float)
        return X
