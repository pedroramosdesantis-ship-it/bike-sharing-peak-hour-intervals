from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--input", default="data/hour.csv")
parser.add_argument("--output", default="outputs_uci")
args=parser.parse_args()
outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
import numpy as np, pandas as pd, math, json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
D=pd.read_csv(args.input)
D['timestamp']=pd.to_datetime(D.dteday)+pd.to_timedelta(D.hr,unit='h')
train=(D.timestamp<'2012-01-01').to_numpy();cal=((D.timestamp>='2012-01-01')&(D.timestamp<'2012-07-01')).to_numpy();test=(D.timestamp>='2012-07-01').to_numpy()
print('splits',train.sum(),cal.sum(),test.sum())
cols=['hr','weekday','mnth','weekend'];D['weekend']=(D.weekday.isin([0,6])).astype(int)
X=D[cols].astype(str);y=D.cnt.to_numpy()
def fit(q,seed):
 return Pipeline([('encoding',ColumnTransformer([('calendar',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cols)])),('model',HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=160,max_leaf_nodes=16,learning_rate=.06,min_samples_leaf=35,l2_regularization=1.,random_state=seed))]).fit(X.loc[train],y[train])
lo,hi=fit(.1,2027),fit(.9,2028)
lcal=lo.predict(X.loc[cal]);ucal=hi.predict(X.loc[cal]);lte=lo.predict(X.loc[test]);ute=hi.predict(X.loc[test]);obs=y[test]
scores=np.maximum.reduce([lcal-y[cal],y[cal]-ucal,np.zeros(cal.sum())])
rcal=D.loc[cal,'hr'].isin([7,8,9,16,17,18,19]).to_numpy();rtest=D.loc[test,'hr'].isin([7,8,9,16,17,18,19]).to_numpy()
def quant(s):return float(np.quantile(s,min(1,math.ceil((len(s)+1)*.8)/len(s)),method='higher'))
qs={'pooled':quant(scores),'rush':quant(scores[rcal]),'nonrush':quant(scores[~rcal])}
models={'Quantile':(np.maximum(0,lte),ute),'Pooled_CQR':(np.maximum(0,lte-qs['pooled']),ute+qs['pooled']),'Group_CQR':(np.maximum(0,lte-np.where(rtest,qs['rush'],qs['nonrush'])),ute+np.where(rtest,qs['rush'],qs['nonrush']))}
rows=[]
for name,(lower,upper) in models.items():
 covered=(obs>=lower)&(obs<=upper)
 for label,mask in [('all',np.ones(len(obs),bool)),('rush',rtest),('nonrush',~rtest)]:
  rows.append({'method':name,'group':label,'n':int(mask.sum()),'coverage':float(covered[mask].mean()),'mean_width':float((upper-lower)[mask].mean())})
M=pd.DataFrame(rows);M.to_csv(outdir/'uci_metrics_reproduced.csv',index=False)
print(M.to_string(index=False));print('quantiles',qs)
