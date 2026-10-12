"""Verified Weizmann masks and deterministic natural/synthetic test cases."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'weizmann-petz'
URL='https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/download/Weizmann_Seg_DB_1obj.ZIP'
ARCHIVE_HASH='c3aa39699a644e9fc58b86b059ac446cba2f66d57fb92e806e61aeb7d7973de1'


def seed_for(name):
    return int(hashlib.sha256(('weizmann-petz-v1:'+name).encode()).hexdigest()[:8],16)


def decode_annotation(a,dominance=128):
    # The supplied PNGs are grayscale images overpainted red, not binary masks.
    # Some have antialiased/compressed red borders. Decode red dominance rather
    # than thresholding the original image intensity or demanding exact RGB.
    a=np.asarray(a,dtype=np.int16)
    return a[:,:,0]-np.maximum(a[:,:,1],a[:,:,2])>dominance


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fetch_archive():
    path=OUT/'data'/'Weizmann_Seg_DB_1obj.ZIP';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        partial=path.with_suffix('.partial')
        with urllib.request.urlopen(URL,timeout=60) as source,partial.open('wb') as target:
            while data:=source.read(1024*1024):target.write(data)
        partial.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ARCHIVE_HASH:
        raise ValueError('Unexpected archive checksum; do not silently change the dataset.')
    return path


def seed_pixels(truth):
    """One deterministic interior seed per available class, background first.

The annotation supplies only seed positions and evaluation labels, never the
Hamiltonian. Pad the mask so an edge-touching class has a finite distance.
"""
    seeds=[]
    for label in (False,True):
        region=truth==label
        if region.any():
            distance=distance_transform_edt(np.pad(region,1))[1:-1,1:-1]
            seeds.append(int(np.argmax(distance)))
    return seeds


def resize_image(a,side):
    # Full-frame square normalization; distortion is documented, not hidden.
    return np.array(Image.fromarray(a.astype(np.float32)).resize((side,side),Image.Resampling.BOX),float)


def resize_mask(a,side):
    return np.array(Image.fromarray(a.astype(np.uint8)).resize((side,side),Image.Resampling.NEAREST),bool)


def save_case(case_id,image,masks,metadata):
    truth=np.sum(masks,axis=0)>=2 if len(masks)==3 else masks[0].astype(bool)
    seeds=seed_pixels(truth);path=OUT/'cases'/f'{case_id}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,image=np.asarray(image,float),truth=truth,masks=masks,seeds=seeds)
    return dict(case_id=case_id,shape=list(image.shape),seeds=seeds,
                foreground_fraction=float(truth.mean()),**metadata)


def prepare():
    archive=fetch_archive();cases=[];source_rows=[];missing=[]
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC check failed.')
        images=sorted(n for n in z.namelist() if '/src_bw/' in n and n.endswith('.png'))
        ids=[n.split('/')[1] for n in images]
        assert len(ids)==100 and len(set(ids))==100
        calibration=set(sorted(ids,key=lambda x:seed_for('split:'+x))[:25])
        for name in images:
            source=name.split('/')[1];split='calibration' if source in calibration else 'test'
            image=np.array(Image.open(io.BytesIO(z.read(name))).convert('L'))
            annotations=sorted(n for n in z.namelist() if n.startswith(f'1obj/{source}/human_seg/') and n.endswith('.png'))
            assert len(annotations)==3
            masks=[];decode_sensitivity=[]
            for annotation in annotations:
                a=np.array(Image.open(io.BytesIO(z.read(annotation))).convert('RGB'))
                red=decode_annotation(a)
                decode_sensitivity.append(float(np.mean(decode_annotation(a,64)!=decode_annotation(a,192))))
                assert red.shape==image.shape and red.any() and not red.all()
                masks.append(red)
            masks=np.stack(masks);truth=masks.sum(axis=0)>=2
            source_rows.append(dict(source_id=source,split=split,shape=list(image.shape),
                annotation_files=annotations,image_file=name,
                image_sha256=hashlib.sha256(z.read(name)).hexdigest(),
                annotation_sha256=[hashlib.sha256(z.read(n)).hexdigest() for n in annotations],
                annotation_decode_64_vs_192_fraction=decode_sensitivity,
                human_disagreement_fraction=float(np.mean((masks.sum(axis=0)>0)&(masks.sum(axis=0)<3)))))
            for side in (8,16):
                scaled=np.stack([resize_mask(m,side) for m in masks])
                cases.append(save_case(f'{source}--square-{side}',resize_image(image,side),scaled,
                    dict(source_id=source,split=split,family='natural-square',category=f'{side}x{side}',
                         preprocessing='full-frame BOX image / NEAREST annotation resize to square')))
            for width in (4,16,64):
                # All valid native-resolution 2xL horizontal windows.
                row_sum=truth[:-1].astype(int)+truth[1:].astype(int)
                cumulative=np.pad(np.cumsum(row_sum,axis=1),((0,0),(1,0)))
                counts=cumulative[:,width:]-cumulative[:,:-width]
                categories={'background':counts==0,'foreground':counts==2*width,
                            'boundary':(counts>=width/2)&(counts<=3*width/2)}
                for category,eligible in categories.items():
                    case_id=f'{source}--strip-{width}-{category}'
                    locations=np.argwhere(eligible)
                    if not len(locations):
                        missing.append(dict(case_id=case_id,reason='no eligible native-resolution window'));continue
                    rng=np.random.default_rng(seed_for(case_id));r,c=locations[rng.integers(len(locations))]
                    cases.append(save_case(case_id,image[r:r+2,c:c+width],masks[:,r:r+2,c:c+width],
                        dict(source_id=source,split=split,family='natural-strip',category=category,
                             width=width,crop_top_left=[int(r),int(c)],eligible_windows=len(locations))))
    # All synthetic cases are tests. No natural-image calibration uses these.
    for side in (8,16):
        yy,xx=np.mgrid[0:side,0:side];x=(xx+.5)/side;y=(yy+.5)/side
        disk=(x-.5)**2+(y-.5)**2<.28**2
        ring=disk&((x-.5)**2+(y-.5)**2>.14**2)
        islands=((x-.25)**2+(y-.5)**2<.15**2)|((x-.75)**2+(y-.5)**2<.15**2)
        bridge=islands|(abs(y-.5)<.55/side)&(x>.25)&(x<.75)
        path=(abs(yy-(side//4+xx//2))<.6)
        for geometry,mask in dict(disk=disk,ring=ring,islands=islands,bridge=bridge,path=path).items():
            for condition in ('clean','noise20','noise60','contrast80','blur'):
                for replicate in (0,1):
                    case_id=f'synthetic--{geometry}-{condition}-{side}-r{replicate}'
                    rng=np.random.default_rng(seed_for(case_id));contrast=80 if condition=='contrast80' else 200
                    image=127.5+contrast*(mask.astype(float)-.5)
                    if condition=='blur':image=gaussian_filter(image,.8)
                    if condition.startswith('noise'):image+=rng.normal(0,int(condition[5:]),image.shape)
                    image=np.clip(image,0,255)
                    # Replicate clean/blur/low-contrast cases with inverted intensities.
                    if replicate==1 and not condition.startswith('noise'):image=255-image
                    cases.append(save_case(case_id,image,mask[None],dict(source_id=case_id,split='synthetic-test',
                        family='synthetic',category=geometry,condition=condition,replicate=replicate)))
    manifest=dict(dataset='Weizmann single-object segmentation + controlled synthetic shapes',
        source_url=URL,archive_sha256=ARCHIVE_HASH,annotation_decode='R-max(G,B)>128; red overlay marks foreground',
        consensus='majority of three human masks',calibration_ids=sorted(calibration),
        test_ids=sorted(set(ids)-calibration),sources=source_rows,cases=cases,missing_cases=missing)
    write_json(ROOT/'weizmann-petz-manifest.json',manifest)
    print(f'Prepared {len(cases)} cases; {len(missing)} unavailable strip categories.',flush=True)
    return manifest


if __name__=='__main__':prepare()
