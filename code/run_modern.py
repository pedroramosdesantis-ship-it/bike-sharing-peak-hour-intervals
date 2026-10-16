"""Capital Bikeshare official-trip-data replication: 2024 train/cal, 2025 test.
Requires: numpy, pandas, scikit-learn, matplotlib. Python >=3.10.
Uses only calendar information known when forecasting; no observed weather.
"""
import argparse, hashlib, io, json, math, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

MONTHS=[f'{year}{month:02d}' for year in (2024,2025) for month in range(1,13)]
BASE='https://s3.amazonaws.com/capitalbikeshare-data/'
TIMESTAMP_COLS=['started_at','start_time','start_date','Start date','Start Date','Start date/time']

def load_month(path, month):
    with zipfile.ZipFile(path) as z:
        items=[x for x in z.namelist() if x.lower().endswith('.csv') and not x.startswith('__MACOSX/')]
        if not items: raise ValueError(f'No CSV in {path}')
        pieces=[]; valid=0; rejected=0; raw=0
        for item in items:
            with z.open(item) as fh:
                for chunk in pd.read_csv(fh, chunksize=100000, low_memory=False):
                    raw+=len(chunk)
                    col=next((c for c in TIMESTAMP_COLS if c in chunk.columns),None)
                    if col is None: raise ValueError(f'No recognized start timestamp in {item}. Columns: {list(chunk.columns)}')
                    dates=pd.to_datetime(chunk[col], errors='coerce', format='mixed')
                    # Monthly archive can contain records slightly outside the labeled month.
                    good=dates.notna() & (dates.dt.strftime('%Y%m')==month)
                    valid+=int(good.sum()); rejected+=int((~good).sum())
                    pieces.append(dates[good].dt.floor('h').value_counts())
        counts=pd.concat(pieces).groupby(level=0).sum() if pieces else pd.Series(dtype='int64')
        return counts, {'month':month,'rows_raw':raw,'rows_accepted':valid,'rows_excluded':rejected,'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def prepare(archive_dir, output_dir, download=False):
    archive_dir.mkdir(parents=True,exist_ok=True);output_dir.mkdir(parents=True,exist_ok=True)
    counts=[]; log=[]
    for month in MONTHS:
        path=archive_dir/f'{month}-capitalbikeshare-tripdata.zip'
        if not path.exists():
            if not download: raise FileNotFoundError(f'Missing {path.name}. Pass --download or place original ZIP in archives directory.')
            url=BASE+path.name
            print('Downloading',url,flush=True)
            try: urllib.request.urlretrieve(url,path)
            except Exception:
                path.unlink(missing_ok=True);raise
        c,detail=load_month(path,month)
        if detail['rows_accepted']<=0: raise RuntimeError(f'Archive has no accepted rides: {path.name}')
        print(month,detail['rows_accepted'],'accepted,',detail['rows_excluded'],'excluded',flush=True)
        counts.append(c);log.append(detail)
    full=pd.date_range('2024-01-01','2025-12-31 23:00:00',freq='h')
    joined=pd.concat(counts).groupby(level=0).sum()
    hourly=pd.DataFrame({'timestamp':full,'cnt':joined.reindex(full,fill_value=0).astype(int).values})
    hourly['hr']=hourly.timestamp.dt.hour
    hourly['weekday']=hourly.timestamp.dt.weekday
    hourly['mnth']=hourly.timestamp.dt.month
    hourly['weekend']=(hourly.weekday>=5).astype(int)
    hourly.to_csv(output_dir/'modern_hourly_2024_2025.csv',index=False)
    (output_dir/'provenance.json').write_text(json.dumps({'source':'https://capitalbikeshare.com/system-data','license_review_required':True,'archives':log,'counts_definition':'number of trip starts by local-clock hour in source CSV (no weather or station restrictions)','total_accepted':int(sum(d['rows_accepted'] for d in log)),'zero_hours':int((hourly.cnt==0).sum()),'expected_hours':len(full)},indent=2),encoding='utf-8')
    return hourly

def quant(scores,alpha=.2):
    scores=np.asarray(scores);n=len(scores)
    return float(np.quantile(scores,min(1,math.ceil((n+1)*(1-alpha))/n),method='higher'))

def analyze(D,output_dir):
    D=D.copy();D['timestamp']=pd.to_datetime(D.timestamp)
    train=(D.timestamp<'2024-09-01').to_numpy();cal=((D.timestamp>='2024-09-01')&(D.timestamp<'2025-01-01')).to_numpy();test=(D.timestamp>='2025-01-01').to_numpy()
    assert min(train.sum(),cal.sum(),test.sum())>1000
    cols=['hr','weekday','mnth','weekend']
    X=D[cols].astype(str);y=D.cnt.to_numpy()
    def fit(q,seed):
        return Pipeline([('encoding',ColumnTransformer([('calendar',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cols)])),('model',HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=160,max_leaf_nodes=16,learning_rate=.06,min_samples_leaf=35,l2_regularization=1.,random_state=seed))]).fit(X.loc[train],y[train])
    lo,hi=fit(.1,2027),fit(.9,2028)
    lcal=lo.predict(X.loc[cal]);ucal=hi.predict(X.loc[cal]);lte=lo.predict(X.loc[test]);ute=hi.predict(X.loc[test]);obs=y[test]
    scores=np.maximum.reduce([lcal-y[cal],y[cal]-ucal,np.zeros(cal.sum())])
    rcal=D.loc[cal,'hr'].isin([7,8,9,16,17,18,19]).to_numpy();rtest=D.loc[test,'hr'].isin([7,8,9,16,17,18,19]).to_numpy()
    qs={'pooled':quant(scores),'rush':quant(scores[rcal]),'nonrush':quant(scores[~rcal])}
    models={'Quantile':(np.maximum(0,lte),ute),'Pooled_CQR':(np.maximum(0,lte-qs['pooled']),ute+qs['pooled']),'Group_CQR':(np.maximum(0,lte-np.where(rtest,qs['rush'],qs['nonrush'])),ute+np.where(rtest,qs['rush'],qs['nonrush']))}
    rows=[];predict=[]
    for name,(lower,upper) in models.items():
        covered=(obs>=lower)&(obs<=upper)
        for label,mask in [('all',np.ones(len(obs),bool)),('rush',rtest),('nonrush',~rtest)]:
            rows.append({'method':name,'group':label,'n':int(mask.sum()),'coverage':float(covered[mask].mean()),'mean_width':float((upper-lower)[mask].mean()),'above_upper':float((obs[mask]>upper[mask]).mean())})
        predict.append(pd.DataFrame({'method':name,'timestamp':D.loc[test,'timestamp'].to_numpy(),'observed':obs,'lower':lower,'upper':upper,'covered':covered.astype(int),'rush':rtest.astype(int)}))
    P=pd.concat(predict,ignore_index=True);P.to_csv(output_dir/'modern_predictions.csv',index=False)
    M=pd.DataFrame(rows);M.to_csv(output_dir/'modern_metrics.csv',index=False)
    pivot=P[P.rush==1].pivot(index='timestamp',columns='method',values=['covered','lower','upper'])
    df=pd.DataFrame({'date':pivot.index.date,'delta_coverage':pivot['covered']['Group_CQR']-pivot['covered']['Pooled_CQR'],'delta_width':(pivot['upper']['Group_CQR']-pivot['lower']['Group_CQR'])-(pivot['upper']['Pooled_CQR']-pivot['lower']['Pooled_CQR'])})
    day_groups=[g[['delta_coverage','delta_width']].to_numpy() for _,g in df.groupby('date')]
    rng=np.random.default_rng(2027); bootstrap=[]
    for _ in range(2000):
        ix=rng.integers(0,len(day_groups),len(day_groups))
        sums=np.sum([day_groups[i].sum(axis=0) for i in ix],axis=0);count=sum(len(day_groups[i]) for i in ix)
        bootstrap.append(sums/count)
    bootstrap=np.asarray(bootstrap)
    report={'train_hours':int(train.sum()),'calibration_hours':int(cal.sum()),'test_hours':int(test.sum()),'score_quantiles':qs,'rush_coverage_difference_pp':float(df.delta_coverage.mean()*100),'rush_coverage_difference_95pct_pp':list((np.quantile(bootstrap[:,0],[.025,.975])*100).round(4)),'rush_width_difference':float(df.delta_width.mean()),'rush_width_difference_95pct':list(np.quantile(bootstrap[:,1],[.025,.975]).round(4)),'caveats':['not paired station-level availability or stockout estimation','post-calibration time shift invalidates exchangeability-based guarantees','2024-2025 modern trip archives are not commensurate with 2011-12 UCI count definitions without adjustment','historical calendar inputs exclude weather forecast availability issues']}
    (output_dir/'modern_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.4))
    subset=M[M.group=='rush'];ax.bar(subset.method,subset.coverage*100);ax.axhline(80,ls='--',c='black',lw=1);ax.set_ylim(0,100);ax.set_ylabel('Observed rush-hour coverage (%)');ax.set_title('Capital Bikeshare 2025 holdout: 80% nominal');fig.tight_layout();fig.savefig(output_dir/'modern_coverage.pdf');plt.close(fig)
    print(M.to_string(index=False));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,default=Path('archives'));p.add_argument('--output',type=Path,default=Path('outputs'));p.add_argument('--download',action='store_true');p.add_argument('--hourly',type=Path,help='Existing aggregated hourly file, bypassing downloads')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    D=pd.read_csv(args.hourly) if args.hourly else prepare(args.archives,args.output,args.download)
    analyze(D,args.output)
