"""Numerical, split, checkpoint and measurement provenance audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from correlation_nn_data import ROOT,OUT,INPUT,init,digest,moments
from weizmann_petz_data import write_json


def repair_ess_file(path):
    meta=json.loads(path.read_text())
    if meta.get('ess_ddof',0)==1:return 0
    npz=path.with_suffix('.npz')
    with np.load(npz) as f:data={k:f[k] for k in f.files}
    total=int(np.prod(data['packed'].shape[:2]));ess=np.minimum(data['ess']*total/(total-1),total)
    data['ess']=ess;off=~np.eye(ess.shape[0],dtype=bool)
    meta['diagnostics']['median_pair_ess']=float(np.median(ess[off]))
    meta['diagnostics']['minimum_pair_ess']=float(ess[off].min());meta['ess_ddof']=1
    np.savez_compressed(npz,**data);meta['measurement_sha256']=digest(npz);write_json(path,meta)
    return 1


def fix_population_ess():
    """Bring in-flight pre-fix ESS diagnostics into the unbiased convention.

    Correlation means, standard errors, split Rhat, and decisions are unchanged.
    The original generator workers finish with their imported code; repair only
    completed outputs after the sampling stage terminates.
    """
    repaired=0;paths=sorted((OUT/'measurements').glob('*/*.json'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for k,count in enumerate(pool.map(repair_ess_file,paths)):
            repaired+=count
            if (k+1)%500==0:print('ESS diagnostic audit',k+1,'/',len(paths),flush=True)
    return repaired


def measurements():
    p=init();assert len(list((OUT/'measurements').glob('*/*.json')))==3600
    repaired=fix_population_ess();quality={};replays=[]
    for rid,r in p['rules'].items():
        ds=[]
        for k,case in enumerate(p['cases']):
            path=OUT/'measurements'/rid/f'{case["case_id"]}.npz';meta=json.loads(path.with_suffix('.json').read_text())
            assert meta['rule']==r and meta['case_id']==case['case_id']
            assert digest(path)==meta['measurement_sha256']
            assert digest(INPUT/f'{case["case_id"]}.npz')==meta['input_sha256']
            data=np.load(path);c=data['corr']
            assert c.shape==(3,256,256) and np.isfinite(c).all() and np.max(abs(c))<=1.000001
            assert np.max(abs(c-c.swapaxes(-1,-2)))<1e-6
            assert np.max(abs(np.diagonal(c,axis1=-2,axis2=-1)-1))<1e-6
            if k==0:
                actual=moments(data['packed'],256)
                for old,new in zip((c,data['se'],data['rhat'],data['ess']),actual):np.testing.assert_allclose(old,new,rtol=3e-6,atol=.002)
                assert np.linalg.eigvalsh(c[-1]).min()>-2e-5
                replays.append(dict(rule_id=rid,case_id=case['case_id'],correlation_max_error=float(abs(c-actual[0]).max())))
            diagnostics=dict(meta['diagnostics']);off=~np.eye(256,dtype=bool)
            diagnostics.update(mean_absolute_correlation=float(abs(c[-1][off]).mean()),
                saturated_pair_fraction=float(np.mean(abs(c[-1][off])>.95)),
                participation_rank=float(256**2/np.sum(c[-1].astype(np.float64)**2)))
            ds.append(diagnostics)
        write_json(OUT/f'diagnostics-{rid}.json',ds)
        quality[rid]=dict(converged=sum(d['converged'] for d in ds),images=len(ds),
            median_pair_ess=float(np.median([d['median_pair_ess'] for d in ds])),mean_failed_pair_fraction=float(np.mean([d['failed_pair_fraction'] for d in ds])),
            mean_absolute_correlation=float(np.mean([d['mean_absolute_correlation'] for d in ds])),
            saturated_pair_fraction=float(np.mean([d['saturated_pair_fraction'] for d in ds])),
            participation_rank=float(np.mean([d['participation_rank'] for d in ds])))
    report=dict(passed=True,files=3600,ess_convention_repaired=repaired,matrix_replays=replays,quality=quality)
    write_json(ROOT/'correlation-nn-measurement-validation.json',report);print('measurement audit passed',flush=True)


def training():
    p=init();fit_paths=list((OUT/'fits').glob('*.json'));ids=[c['source_id'] for c in p['cases']]
    cache_checks=0
    for rid in p['rules']:
        consolidated=np.load(OUT/f'correlations-{rid}.npy',mmap_mode='r')
        cache=np.load(OUT/'training-cache'/f'{rid}.npy',mmap_mode='r')
        for i,case in enumerate(p['cases']):
            c=np.load(OUT/'measurements'/rid/f'{case["case_id"]}.npz')['corr']
            np.testing.assert_array_equal(c,consolidated[i])
            np.testing.assert_array_equal(c[-1],cache[i]);cache_checks+=1
    for path in fit_paths:
        m=json.loads(path.read_text());f=p['folds'][m['job']['fold']]
        assert m['train_ids']==f['train'] and m['validation_ids']==f['validation'] and m['test_ids_never_read']==f['test']
        assert not(set(m['train_ids'])&set(f['test']) or set(m['validation_ids'])&set(f['test']))
        assert digest(path.with_suffix('.pt'))==m['checkpoint_sha256']
        v=np.load(path.with_suffix('.npz'));assert v['validation_indices'].tolist()==[ids.index(s) for s in f['validation']]
    selection=json.loads((ROOT/'correlation-nn-final-selection.json').read_text());final_paths=list((OUT/'final').glob('*.json'))
    assert len(final_paths)==3*len(selection)
    for path in final_paths:
        m=json.loads(path.read_text());s=selection[m['name']];f=p['folds'][m['selection_job']['fold']]
        assert m['selection_job']==s['job'] and m['selected_threshold']==s['result']['threshold']
        assert set(m['train_ids'])==set(f['train']+f['validation']) and m['test_ids']==f['test']
        assert not set(m['train_ids'])&set(m['test_ids'])
        assert digest(path.with_suffix('.pt'))==m['checkpoint_sha256']
        a=np.load(path.with_suffix('.npz'));assert a['test_indices'].tolist()==[ids.index(i) for i in f['test']]
        assert a['probabilities'].shape==(20,256) and np.isfinite(a['probabilities']).all()
        assert ((a['probabilities']>=0)&(a['probabilities']<=1)).all()
    write_json(ROOT/'correlation-nn-training-validation.json',dict(passed=True,inner_fits=len(fit_paths),final_fits=len(final_paths),
        evaluation_images=100,outer_folds=5,replicates=3,no_source_overlap=True,checkpoint_hashes_checked=True,measurement_to_training_cache_checks=cache_checks))
    print('training audit passed:',len(fit_paths),'inner fits;',len(final_paths),'refits',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['measurements','training']);a=ap.parse_args()
    (measurements if a.stage=='measurements' else training)()
