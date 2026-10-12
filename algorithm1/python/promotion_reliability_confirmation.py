"""Execution amendment: stop after the largest requested confirmation cap.

The original ten-round child-seed spawning schedule is retained. No state or
observable used by any selected policy/control is changed.
"""
import json
import run_promotion_reliability as main
AMEND=main.ROOT/'promotion-reliability-execution-amendment.json'
CAPS={main.identifier(g['rule']):max(p['cap'] for rows in g['profiles'].values() for p in rows.values()) for g in main.grouped('test')}
_original_tail=main.iter_tail
_original_bank=main.bank


def limited_tail(image,rule,**kwargs):
    limit=CAPS[main.identifier(rule)] if kwargs['samples']==main.SETTINGS['confirm']['samples'] else 10
    if kwargs['start_round']>=limit:return
    for i,value in enumerate(_original_tail(image,rule,**kwargs),kwargs['start_round']):
        yield value
        if i+1>=limit:break


def amended_bank(stage,c,rule):
    packed,meta,data=_original_bank(stage,c,rule)
    if len(packed)<10:
        assert stage=='confirm' and len(packed)>=CAPS[main.identifier(rule)]
        amendment=dict(sha256=main.digest(AMEND),executed_with_maximum_requested_cap=CAPS[main.identifier(rule)])
        if 'execution_amendment' in meta:assert meta['execution_amendment']==amendment
        else:
            meta['execution_amendment']=amendment
            main.write_json(main.OUT/'banks'/stage/f'{c["case_id"]}--{main.identifier(rule)}.json',meta)
    return packed,meta,data


def freeze():
    v=dict(date='2026-10-09',reason='Avoid sampling unused future rounds in longer confirmations. All policies for the secondary g/D=.5 Hamiltonian have cap3; the other settings still require10.',
        maximum_requested_caps=CAPS,base_code_hashes=main.hashes(),source_sha256=main.digest(__file__),
        invariant='Keep rounds=10 for SeedSequence.spawn and uncertainty correction, but stop the sampler generator after the largest cap in all selected profiles and controls for that Hamiltonian. Do not truncate any pre-existing prefix. No outcomes or diagnostics are used to choose this limit. Policy traces and equal-budget controls must equal the full-bank prefix.')
    if AMEND.exists():assert json.loads(AMEND.read_text())==v
    else:main.write_json(AMEND,v)


main.iter_tail=limited_tail
main.bank=amended_bank
if __name__=='__main__':freeze();main.main()
