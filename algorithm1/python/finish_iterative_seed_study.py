"""Continue the authorized experiment after calibration, preserving logs.

This is a one-off sequential process, not a scheduled automation. Scientific
settings remain in the frozen runner. It can resume completed stages safely.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'algorithm1/results/iterative-seeds'
STATUS=OUT/'execution-status.json'


def status(stage,**more):
    value=dict(stage=stage,updated_epoch=time.time(),**more)
    STATUS.parent.mkdir(parents=True,exist_ok=True);STATUS.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


def main():
    while True:
        count=sum(not p.name.endswith('.bank.json') for p in (OUT/'calibrate').glob('*.json'))
        if count==4800:break
        if count>4800:raise RuntimeError('Unexpected calibration record count.')
        log=Path('/private/tmp/iterative-seeds-calibration.log')
        if log.exists() and 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Calibration reported an exception; inspect its log.')
        status('waiting_for_calibration',complete=count,total=4800)
        time.sleep(30)
    commands=[
        ('select',['algorithm1/python/run_iterative_seed_study.py','select']),
        ('test',['algorithm1/python/run_iterative_seed_extended.py','test','--workers','4']),
        ('confirm',['algorithm1/python/run_iterative_seed_extended.py','confirm','--workers','4']),
        ('analysis',['algorithm1/python/analyze_iterative_seed_study.py']),
        ('audit',['algorithm1/python/validate_iterative_seed_study.py'])]
    for stage,args in commands:
        logfile=Path('/private/tmp')/f'iterative-seeds-{stage}.log';status(stage,log=str(logfile))
        with logfile.open('w') as f:subprocess.run([sys.executable,*args],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
    status('completed')


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error));raise
