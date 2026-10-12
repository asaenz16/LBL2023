"""Finish this finite experiment once the comparison files are complete."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/higher-seeds'


def main():
    start=time.monotonic();last=-1
    while True:
        files=list((OUT/'test').glob('*.json'))
        if len(files)!=last:
            print('comparison files:',len(files),'/1800',flush=True);last=len(files)
        if len(files)==1800:
            try:
                for p in files:json.loads(p.read_text())
            except json.JSONDecodeError:
                time.sleep(2);continue
            break
        if time.monotonic()-start>10800:raise RuntimeError('Comparison not complete after three hours; inspect run logs.')
        time.sleep(10)
    for name,args in [('audit_higher_seed_study.py',['controls','--workers','3']),
                      ('audit_higher_seed_study.py',['audit']),
                      ('analyze_higher_seed_study.py',[])]:
        print('starting',name,*args,flush=True)
        subprocess.run([sys.executable,str(ROOT/'python'/name),*args],check=True)
    print('COMPLETE: metrics, validation, report and figures written',flush=True)


if __name__=='__main__':main()
