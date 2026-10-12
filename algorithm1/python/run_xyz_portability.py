"""Post-selection flat-image diagnostic; does not tune the Weizmann experiment."""
import json
import numpy as np
from run_weizmann_xyz import ROOT,PRIMARY_BUDGET
from xyz_alignment import parameterize,thermal_distributions,correlations,draw_correlations,metrics,uncertainty_radius
from weizmann_petz_data import seed_pixels,write_json,seed_for


def main():
    selected=json.loads((ROOT/'weizmann-xyz-selection.json').read_text())['winners']
    image=np.tile([0.,0.,255.,255.],(2,1));truth=image>125;seeds=seed_pixels(truth);records=[]
    for label in ['Ising_flexible','XYZ_staggered_flexible','XYZ_fixed_triple_diagnostic']:
        entry=selected['XYZ_staggered_flexible' if label.startswith('XYZ') else 'Ising_flexible'];rule=entry['rule'].copy()
        if label=='XYZ_fixed_triple_diagnostic':rule['threshold']='fixed_triple';rule['width']=0.
        edges,j,param=parameterize(image,rule);p,_=thermal_distributions(image.size,edges,j,[param['b']]);axis=correlations(p[0],seeds)
        target=axis.mean(axis=0) if entry['observable']=='alignment' else axis[2]
        measured=[]
        for rep in range(16):
            c=draw_correlations(p[0],seeds,PRIMARY_BUDGET,np.random.default_rng(seed_for('xyz-portability:'+label+str(rep))),entry['observable'])
            measured.append(metrics(c,truth,seeds,entry['decision'],uncertainty_radius(image.size,PRIMARY_BUDGET,entry['decision'])))
        row=dict(label=label,rule=rule,parameters=param,exact=metrics(target,truth,seeds,entry['decision']),sample={k:float(np.mean([r[k] for r in measured])) for k in measured[0]})
        records.append(row)
        print(label,'thresholds',param['thresholds'],'exact IoU',row['exact']['iou'],'unresolved',1-row['sample']['resolved'])
    write_json(ROOT/'weizmann-xyz-portability.json',dict(note='Post-selection diagnostic; no parameter selection or Weizmann test retuning. Single noiseless 2x4 two-block image; not evidence of general performance.',image=image.tolist(),truth=truth.astype(int).tolist(),seeds=seeds,budget=PRIMARY_BUDGET,records=records))


if __name__=='__main__':main()
