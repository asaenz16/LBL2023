"""Execution-only optimization: skip rounds beyond every requested endpoint.

The frozen mathematical implementation, ten-round confidence correction,
random priorities, bank prefixes, and returned results are unchanged. This
wrapper is recorded separately as an execution amendment and replay-tested
against the original evaluator.
"""
import higher_seed_readout as h
import run_higher_seed_study as runner

ORIGINAL_EVALUATE = runner.evaluate


def bounded_evaluate(*args, **kwargs):
    policies=args[5] if len(args)>5 else kwargs['policies']
    caps={}
    for p in policies:
        key=tuple(sorted((k,v) for k,v in p.items() if k not in ('cap','stopping')))
        caps[key]=max(caps.get(key,0),p['cap'])
    original=h.trajectory
    def clipped(evidence,initial,start,config,uniforms):
        cap=caps[tuple(sorted(config.items()))]
        return original(evidence[:cap],initial,start,config,uniforms)
    h.trajectory=clipped
    try:return ORIGINAL_EVALUATE(*args,**kwargs)
    finally:h.trajectory=original


runner.evaluate=bounded_evaluate


if __name__=='__main__':runner.main()
