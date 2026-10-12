"""Pre-test matched-function extension, using calibration decisions only."""
import hashlib
import json
from pathlib import Path
from run_weizmann_ising_seeds import ROOT,FUNCTIONS,hashes,identifier
from weizmann_petz_data import write_json


def main():
    selection_path=ROOT/'weizmann-ising-seeds-selection.json';selection=json.loads(selection_path.read_text())
    control_path=ROOT/'weizmann-ising-seeds-controls-protocol.json'
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if 'matched_control_extension' in selection:
        old=json.loads(control_path.read_text())
        assert old['extension_code_sha256']==source_hash
        assert old['expanded_selection']==selection
        print('Existing matched-control selection verified.');return
    if selection['code_hashes']!=hashes():raise ValueError('Selection code changed.')
    original=json.loads(json.dumps(selection));base=selection['winners']['arctan__two_max']['rule']
    records=json.loads((ROOT/'weizmann-ising-seeds-calibration.json').read_text())['scores']
    for fn in FUNCTIONS:
        r=dict(base,function=fn)
        if fn=='linear':r['sensitivity']='adaptive'
        selection['winners']['matched__'+fn]=next(x for x in records if x['rule']==r and x['decision']=='two_max')
    selection['matched_control_extension']='weizmann-ising-seeds-controls-protocol.json'
    protocol=dict(date='2026-10-08',extension_code_sha256=source_hash,original_selection=original,expanded_selection=selection,
        timing='Declared after launching calibration and before any new-study held-out tests. No test results are used.',
        purpose='Change only the coupling function on the arctan two-seed calibration winner. Retain threshold, K and sensitivity. Pure linear uses its canonical alpha because median normalization removes overall alpha. Actual b changes to retain the same median dimensionless interaction strength.',
        original_protocol='weizmann-ising-seeds-protocol.json')
    if control_path.exists() and json.loads(control_path.read_text())!=protocol:raise ValueError('Matched-control protocol changed.')
    write_json(control_path,protocol);write_json(selection_path,selection)
    print('Added five matched-function labels; duplicate Hamiltonians share all samples.')


if __name__=='__main__':main()
