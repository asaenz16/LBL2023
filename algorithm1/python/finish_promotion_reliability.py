"""Sequential continuation of this experiment after all comparison files exist."""
import os
for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/promotion-reliability'
while len(list((OUT/'test').glob('*.json')))<750:time.sleep(10)
steps=[('confirmation',['promotion_reliability_confirmation.py','confirm','--workers','4']),
    ('analysis',['analyze_promotion_reliability.py']),('audit',['audit_promotion_reliability.py']),
    ('report',['write_promotion_reliability_report.py'])]
for label,args in steps:
    print('Starting',label,flush=True)
    with (OUT/f'{label}.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'python'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Finished',label,flush=True)
