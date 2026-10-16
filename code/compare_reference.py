from pathlib import Path
import argparse,json,sys
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--reproduced',type=Path,required=True);a=p.parse_args()
passes=[]
for name in ['modern_predictions.csv','modern_metrics.csv']:
 x=pd.read_csv(a.reference/name);y=pd.read_csv(a.reproduced/name);ok=x.equals(y);print(('PASS' if ok else 'FAIL'),name,'rows',len(x));passes.append(ok)
name='modern_report.json';x=json.loads((a.reference/name).read_text());y=json.loads((a.reproduced/name).read_text());ok=x==y;passes.append(ok);print(('PASS' if ok else 'FAIL'),name)
sys.exit(0 if all(passes) else 1)
