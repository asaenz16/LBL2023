"""Apply the documented loop-limit amendment while preserving frozen settings."""
import json
import iterative_seed_gibbs as gibbs
import run_iterative_seed_study as runner
from iterative_seed_extended_loops import _snapshots_extended

_original_snapshots=gibbs._snapshots
_original_job=runner.job
AMENDMENT=runner.ROOT/'iterative-seeds-numerical-amendment.json'


def amended_job(spec):
    stage,c,rule,policies=spec
    path=runner.OUT/stage/f'{c["case_id"]}--{runner.identifier(rule)}.json'
    amendment=json.loads(AMENDMENT.read_text())
    use_extended=stage!='calibrate' or path.stem==amendment['failed_case']
    gibbs._snapshots=_snapshots_extended if use_extended else _original_snapshots
    fresh=not path.exists() and not path.with_suffix('.bank.json').exists()
    result=_original_job(spec)
    if fresh and use_extended:
        assert amendment['base_code_hashes']==runner.hashes()
        assert amendment['extended_sampler_sha256']==runner.digest(runner.ROOT/'python/iterative_seed_extended_loops.py')
        for p in (path,path.with_suffix('.bank.json')):
            row=json.loads(p.read_text());row['execution_amendment']=dict(
                amendment_sha256=runner.digest(AMENDMENT),extended_sampler_sha256=amendment['extended_sampler_sha256'])
            runner.write_json(p,row)
    return result


runner.job=amended_job

if __name__=='__main__':runner.main()
