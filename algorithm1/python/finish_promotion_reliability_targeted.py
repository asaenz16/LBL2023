"""Finish the supplementary analysis once its fixed comparison run completes."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/promotion-reliability';TARGET=OUT/'reliability-targeted'
while len([p for p in TARGET.glob('*.json') if p.name!='case-summary.json'])<450 or not (ROOT/'promotion-reliability.md').exists():time.sleep(10)
with (OUT/'targeted-analysis.log').open('w') as log:
    for name in ('analyze_promotion_reliability_targeted.py','append_promotion_reliability_targeted_report.py'):
        subprocess.run([sys.executable,str(ROOT/'python'/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
print('Supplementary analysis, audit and report complete',flush=True)
