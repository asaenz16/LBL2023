"""Finite, resumable continuation of the requested experiment."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/correlation-nn'


def run(script,*args):
    name=script+'-'+ '-'.join(args)
    with (OUT/(name+'.log')).open('a') as log:
        print('START',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'python'/f'{script}.py'),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,flush=True)


def main():
    started=time.time()
    while True:
        p=json.loads((ROOT/'correlation-nn-protocol.json').read_text())
        sampling_done=all((OUT/f'correlations-{rid}.npy').exists() and (OUT/f'diagnostics-{rid}.json').exists() for rid in p['rules'])
        jobs_path=OUT/'screen-jobs.json'
        screen_done=jobs_path.exists() and all((OUT/'fits'/f'{j}.json').exists() for j in json.loads(jobs_path.read_text()))
        if sampling_done and screen_done:break
        for path in [ROOT/'results/correlation-nn-sampling.log',OUT/'training-screen.log']:
            if path.exists() and 'Traceback (most recent call last)' in path.read_text():raise RuntimeError('Upstream failed: '+str(path))
        if time.time()-started>12*3600:raise TimeoutError('Upstream stages did not complete.')
        time.sleep(5)
    run('audit_correlation_nn','measurements')
    run('run_correlation_nn','refine','--workers','6')
    run('run_correlation_nn','select')
    with (OUT/'confirmation-sampling.log').open('a') as log:
        confirm=subprocess.Popen([sys.executable,str(ROOT/'python/confirm_correlation_nn.py'),'sample','--scope','both','--workers','2'],stdout=log,stderr=subprocess.STDOUT)
        run('run_correlation_nn','columns','--workers','4')
        run('run_correlation_nn','final','--workers','4')
        if confirm.wait()!=0:raise RuntimeError('Confirmation sampling failed.')
    run('audit_correlation_nn','training')
    run('analyze_correlation_nn')
    run('confirm_correlation_nn','infer')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
