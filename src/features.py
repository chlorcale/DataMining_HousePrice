"""Optional pre-sale feature engineering: never reads sale price or sale date.

All transforms here are row-wise; imputation/encoding belongs INSIDE each
member's training-fold pipeline. Do not silently fill absent source fields.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def col(df, name):
    if name in df:
        return pd.to_numeric(df[name], errors='coerce')
    return pd.Series(np.nan, index=df.index)

class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, engineered=True):
        self.engineered=engineered
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X=X.copy()
        if not self.engineered:
            return X
        living=col(X,'gr_liv_area')
        quality=col(X,'overall_qual')
        X['quality_x_living_area']=quality*living
        X['total_bathrooms']=col(X,'full_bath')+0.5*col(X,'half_bath')+col(X,'bsmt_full_bath')+0.5*col(X,'bsmt_half_bath')
        X['porch_total']=sum(col(X,c).fillna(0) for c in ('open_porch_sf','enclosed_porch','screen_porch','3ssn_porch'))
        # Includes UNFINISHED basement: do not call this 'finished area'.
        X['living_plus_total_basement_sf']=living+col(X,'total_bsmt_sf').fillna(0)
        X['garage_presence']=col(X,'garage_cars').gt(0).astype(int)
        return X
