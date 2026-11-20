from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import read_dataset,summarize
X,y=read_dataset(); print(summarize(X,y)); assert len(y)==2930 and X.shape[1]==79
print('PASS: original Ames, 2930 rows, 79 predictors')
