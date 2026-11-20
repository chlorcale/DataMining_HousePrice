"""Frozen data split and CV protocols."""
import numpy as np
from sklearn.model_selection import train_test_split, KFold

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_SEED = 123  # Original ULTRA baseline; separate advanced experiments used 2026.

def holdout_indices(n_rows):
    indices = np.arange(n_rows)
    return train_test_split(indices, test_size=TEST_SIZE, random_state=SEED)

def make_cv():
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
