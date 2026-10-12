"""Prespecified longer/high-shot checks; no held-out score-based selection."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from run_weizmann_petz import manifest,benchmark_job,DEFAULT,ROOT


def main():
    data=manifest();by_id={c['case_id']:c for c in data['cases']}
    chosen=json.loads((ROOT/'weizmann-petz-calibration.json').read_text())['selected']
    specs=[]
    # First two calibration IDs, never chosen using test-set performance.
    selections=[(s+'--square-16','calibrated',chosen) for s in data['calibration_ids'][:2]]
    selections += [('synthetic--bridge-clean-16-r0','original',DEFAULT),
                   ('synthetic--islands-noise60-16-r0','original',DEFAULT)]
    for case_id,name,params in selections:
        for initial in ('plus_x','all_up'):
            specs.append((by_id[case_id],name,params,16384,[0,20,50,200,500,1000,2000],initial,'long'))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(benchmark_job,s) for s in specs]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();print(json.dumps(dict(done=i,total=len(specs),case=r['case']['case_id'],
                initial=r['signature']['initial'],final=r['records'][-1])),flush=True)


if __name__=='__main__':main()
