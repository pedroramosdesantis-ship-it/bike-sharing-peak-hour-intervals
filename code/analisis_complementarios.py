"""Deterministic diagnostics from the 2025 holdout predictions. No refitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--predictions',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('supplemental'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
P=pd.read_csv(a.predictions,parse_dates=['timestamp'])
assert P.shape[0]==26280
assert set(P.method)=={'Quantile','Pooled_CQR','Group_CQR'}
peak=P[P.rush==1].copy();peak['month']=peak.timestamp.dt.month
monthly=peak.groupby(['month','method'],sort=True).agg(n=('covered','size'),coverage=('covered','mean'),mean_width=('upper',lambda v:np.nan)).reset_index()
# Recompute mean width without aggregation ambiguity
width=peak.assign(width=peak.upper-peak.lower).groupby(['month','method']).width.mean().reset_index(name='mean_width')
monthly=monthly.drop(columns='mean_width').merge(width,on=['month','method'])
monthly.to_csv(a.output/'monthly_peak_coverage.csv',index=False)
P['width']=P.upper-P.lower
P['upper_exceedance']=np.maximum(P.observed-P.upper,0)
errors=P[P.rush==1].groupby('method').agg(upper_exceedance_mean_all_peak=('upper_exceedance','mean'),upper_miss_rate=('upper_exceedance',lambda v:(v>0).mean()),width=('width','mean'))
errors.to_csv(a.output/'upper_exceedance.csv')
# Dates affected by naive wall-clock DST convention in the 2025 test sample
problem={pd.Timestamp('2025-03-09 02:00:00'),pd.Timestamp('2025-11-02 01:00:00')}
P['dst_affected']=P.timestamp.isin(problem)
sens=P.groupby('method').agg(n=('covered','size'),coverage=('covered','mean'))
sens_clean=P[~P.dst_affected].groupby('method').agg(n_without_dst=('covered','size'),coverage_without_dst=('covered','mean'))
sens=sens.join(sens_clean);sens.to_csv(a.output/'dst_sensitivity.csv')
# Illustrative upper-endpoint planning loss, calculated from observed trip starts
for k in [1,5,5.38,10]:
 peak[f'loss_lambda_{k}']=np.maximum(peak.upper-peak.observed,0)+k*np.maximum(peak.observed-peak.upper,0)
cost=peak.groupby('method')[[c for c in peak if c.startswith('loss_lambda_')]].mean()
cost.to_csv(a.output/'hypothetical_loss_scenarios.csv')
# Root where pooled and groupwise hypothetical peak-hour loss equal
x=peak.groupby('method').apply(lambda d:pd.Series({'over':np.maximum(d.upper-d.observed,0).mean(),'under':np.maximum(d.observed-d.upper,0).mean()}),include_groups=False)
pool=x.loc['Pooled_CQR'];group=x.loc['Group_CQR']
intersection=(group.over-pool.over)/(pool.under-group.under)
(a.output/'intersection.txt').write_text(f'Calculated switch penalty lambda = {intersection:.9f}\n',encoding='utf-8')
print('Monthly diagnostics saved; peak upper-exceedance means:',errors.to_dict(orient='index'))
print('Hypothetical penalty crossing:',intersection)
