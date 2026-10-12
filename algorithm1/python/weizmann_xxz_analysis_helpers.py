"""Selection-dependent comparison settings for the cluster-sampler study."""
import json
from run_weizmann_xxz_v2 import ROOT


def scan_rules():
    selected=json.loads((ROOT/'weizmann-xxz-v2-selection.json').read_text())
    base=selected['overall']['XXZ']['rule']
    rules=[dict(base,transverse='relative',strength=g) for g in (0.,.25,.5,1.,2.)]
    rules.append(dict(base,transverse='absolute',strength=1.,temperature='combined'))
    return rules
