"""
These are *unfitted factories*, designed to go INSIDE CV model pipelines.
Exact FeatureBuilder is preserved separately in src/features.py.
No target, per-fold validation data, or held-out test is used in fit.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from .features import FeatureBuilder

class LightGBMFrame(BaseEstimator, TransformerMixin):
    """Exact categorical fitting semantics from ULTRA src/advanced.py."""
    def fit(self, X, y=None):
        self.cols_ = X.columns.tolist()
        self.cats_ = {
            c: list(X[c].dropna().astype(str).unique())
            for c in X if not pd.api.types.is_numeric_dtype(X[c])
        }
        return self

    def transform(self, X):
        z = X.reindex(columns=self.cols_).copy()
        for c, v in self.cats_.items():
            z[c] = pd.Categorical(z[c].astype('string'), categories=v)
        for c in z.columns:
            if c not in self.cats_:
                z[c] = pd.to_numeric(z[c], errors='coerce').astype('float32')
        return z


def make_linear_preprocessor(engineered=True):
    """Original ULTRA Ridge/ElasticNet impute, OHE and numeric scaling."""
    numeric = make_column_selector(dtype_include=np.number)
    categorical = make_column_selector(dtype_exclude=np.number)
    num = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    cat = Pipeline([
        ('fill', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    pre = ColumnTransformer([('num', num, numeric), ('cat', cat, categorical)],
                            remainder='drop', sparse_threshold=0)
    return Pipeline([('feature_builder', FeatureBuilder(engineered=engineered)), ('pre', pre)])


def make_tree_preprocessor(engineered=True):
    """Original ULTRA RandomForest feature processing, without estimator."""
    numeric = make_column_selector(dtype_include=np.number)
    categorical = make_column_selector(dtype_exclude=np.number)
    num = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
    ])
    cat = Pipeline([
        ('fill', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
        ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                 encoded_missing_value=-1)),
    ])
    pre = ColumnTransformer([('num', num, numeric), ('cat', cat, categorical)],
                            remainder='drop', sparse_threshold=0)
    return Pipeline([('feature_builder', FeatureBuilder(engineered=engineered)), ('pre', pre)])


def make_lightgbm_preprocessor(engineered=True):
    """Original ULTRA LightGBM path: engineered features + native categorical.

    Missing numeric values are intentionally NaN: native LightGBM learns their handling.
    Categorical levels are learned on fit partition only, unseen validation levels -> NaN.
    """
    return Pipeline([('feature_builder', FeatureBuilder(engineered=engineered)),
                     ('frame', LightGBMFrame())])


def make_preprocessor(algorithm, engineered=True):
    name = algorithm.lower().replace('_', '').replace('-', '')
    if name in ('ridge', 'elasticnet'):
        return make_linear_preprocessor(engineered)
    if name in ('rf', 'randomforest'):
        return make_tree_preprocessor(engineered)
    if name in ('lightgbm', 'lgbm'):
        return make_lightgbm_preprocessor(engineered)
    raise ValueError(f'Unknown algorithm: {algorithm}')
