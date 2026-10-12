"""Exchange scan with a documented temperature correction for absolute g=1.

Relative-strength comparisons keep the selected temperature rule. The
absolute-strength comparator uses combined normalization, as in calibration,
to avoid an unintended hundreds-fold increase in dimensionless exchange.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from run_weizmann_xxz import ROOT,INPUT,manifest,dispatch,hashes
from xxz_image_gibbs import parameterize
from weizmann_petz_data import write_json


def scan_rules():
    selection=json.loads((ROOT/'weizmann-xxz-selection.json').read_text())
    base=selection['overall']['XXZ']['rule']
    rules=[dict(base,transverse='relative',strength=g) for g in (0.,.25,.5,1.,2.)]
    rules.append(dict(base,transverse='absolute',strength=1.,temperature='combined'))
    return rules


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    rules=scan_rules();cases=[c for c in manifest()['cases'] if c['split']=='test'
        and c['family']=='natural-square' and c['shape']==[8,8]]
    bad=[]
    for c in cases:
        f=np.load(INPUT/'cases'/f'{c["case_id"]}.npz')
        _,p=parameterize(f['image'],dict(rules[-1],temperature='longitudinal'))
        bad.append(p['b']*p['g'])
    amendment=dict(date='2026-10-08',original_protocol='weizmann-xxz-protocol.json',
        change='Only the absolute g=1 arm of the mechanism scan uses combined temperature normalization, matching its calibration definition. All relative-g arms and primary comparisons are unchanged.',
        reason='At the selected small alpha, retaining longitudinal beta while setting g=1 produces very large b*g and excessive SSE expansion/loop work. Two unfinished absolute-g jobs were stopped; no absolute-g result was used to choose this change.',
        aborted_absolute_longitudinal_bg_range=[min(bad),max(bad)],
        absolute_combined_bg_range='b*g <= 4K = 4.8',
        scan_rules=rules,code_hashes=hashes(),scan_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        additional_controls='Apply the already-planned three matched-b classical controls to the 75 native 2x4 test crops as well as the squares; also report all previously selected Ising family rules on those crops, with no retuning.')
    path=ROOT/'weizmann-xxz-amendment.json'
    if path.exists() and json.loads(path.read_text())!=amendment:raise ValueError('Changed scan amendment.')
    write_json(path,amendment)
    dispatch([('scan',c,r) for c in cases for r in rules],args.workers)


if __name__=='__main__':main()
