"""Exact classical ablations at the selected XXZ model's same temperature."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from run_weizmann_xxz import OUT,ROOT,INPUT,manifest,identifier,hashes
from xxz_image_gibbs import parameterize,evaluate
from ising_grid_transfer import exact_grid_gibbs
from weizmann_petz_data import write_json


def isolated_bond_shift(b,g):
    # log(cosh(x)) evaluated without exponential overflow.
    x=b*g/2
    return 2/b*(np.logaddexp(x,-x)-np.log(2.))


def controls(image,rule,seeds,truth):
    model,p=parameterize(image,rule);g=p['g'];b=p['b'];shift=isolated_bond_shift(b,g)
    choices=dict(remove_exchange=model.delta.copy(),
        neutral_band=np.where(abs(model.delta)<=g,0.,model.delta),
        isolated_bond=model.delta-shift)
    result={};arrays={}
    for name,delta in choices.items():
        changed=replace(model,delta=delta)
        ref=exact_grid_gibbs(SimpleNamespace(graph=changed,b=b,n=model.n),seeds)
        result[name]=evaluate(ref,0.,truth,seeds);arrays[name]=ref
    return result,arrays,dict(**p,isolated_bond_shift=shift)


def main():
    selected=json.loads((ROOT/'weizmann-xxz-selection.json').read_text());rule=selected['overall']['XXZ']['rule']
    rows=[];cases=[c for c in manifest()['cases'] if c['split']=='test' and (c['family']=='natural-square'
        or (c['family']=='natural-strip' and c['width']==4 and c['category']=='boundary'))]
    script_hash=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    for case in cases:
        f=np.load(INPUT/'cases'/f'{case["case_id"]}.npz')
        metrics,arrays,p=controls(f['image'],rule,f['seeds'],f['truth'])
        # Already-selected Ising family winners, with no crop-specific retuning.
        if case['family']=='natural-strip':
            for name,winner in selected['winners'].items():
                if name.startswith('Ising/'):
                    m,q=parameterize(f['image'],winner['rule'])
                    ref=exact_grid_gibbs(SimpleNamespace(graph=m,b=q['b'],n=m.n),f['seeds'])
                    metrics[name]=evaluate(ref,0.,f['truth'],f['seeds']);arrays[name.replace('/','_')]=ref
        rows.append(dict(case=case,rule=rule,parameters=p,metrics=metrics))
        path=OUT/'controls'/f'{case["case_id"]}.npz';path.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(path,**arrays)
    write_json(ROOT/'weizmann-xxz-controls.json',dict(rows=rows,code_hashes=hashes(),control_code_sha256=script_hash,
        interpretation='Classical diagnostic Hamiltonians, not alternative claims to simulate XXZ. Parameters and b are matched without retuning.'))
    print('Saved',len(rows),'matched-temperature classical control triplets.',flush=True)


if __name__=='__main__':main()
