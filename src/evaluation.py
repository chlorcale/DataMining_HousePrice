import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float)
 return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p)),'mape_pct':float(np.mean(np.abs((y-p)/np.maximum(y,1)))*100),'rmsle':float(np.sqrt(np.mean((np.log1p(np.maximum(y,0))-np.log1p(np.maximum(p,0)))**2)))}
