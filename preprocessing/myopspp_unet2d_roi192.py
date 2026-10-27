"""Export two fixed 192-mm MyoPS++ ROIs from the locked U-Net localizer.

No model training, prediction-center changes, or writes to existing datasets.
Images: bilinear, labels: nearest-exact, native Z retained. Exported artifacts
are NIfTI and provenance, not SCAR NPZ/H5 training caches.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F

from preprocessing.myopspp_unet2d import (
    RAW_CODES, RUN, crop_fixed, digest, label_counts, normalize,
    predicted_center, resample_xy, save_nii, target_grid, write_json,
)

FOV = 192
WORKSPACE = Path(__file__).resolve().parents[1]
REPORT = WORKSPACE / 'output' / 'roi192_export_2026-10-27'
DESTINATIONS = {
    128: Path(r'D:\NCKH\MyoPSpp_UNet2D_ROI192mm_128px_1p5mm_2026-10-27'),
    192: Path(r'D:\NCKH\MyoPSpp_UNet2D_ROI192mm_192px_1mm_2026-10-27'),
}
OLD_EXPORTS = (
    Path(r'D:\NCKH\MyoPSpp_UNet2D_ROI128_2026-10-27'),
    Path(r'D:\NCKH\MyoPSpp_UNet2D_ROI128_1mm_2026-10-27'),
)
MODALITIES = ('C0', 'LGE', 'T2')
GROUPS = {
    'myocardium': (200, 1220, 2221), 'pathology': (1220, 2221),
    'scar': (2221,), 'edema_exclusive': (1220,), 'LV': (500,), 'RV': (600,),
    'all_labels': (200, 500, 600, 1220, 2221),
}


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def resize_to(array, affine, side, is_label=False):
    """Preserve pixel-edge FOV when changing the square in-plane matrix."""
    if array.ndim != 3 or array.shape[0] != array.shape[1] or side <= 0:
        raise ValueError('Expected nonempty square HWD ROI and positive output side')
    if array.shape[:2] == (side, side):
        return array, np.array(affine, copy=True)
    source_side = array.shape[0]
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32))[:, None]
    if is_label:
        tensor = F.interpolate(tensor, size=(side, side), mode='nearest-exact')
    else:
        tensor = F.interpolate(tensor, size=(side, side), mode='bilinear', align_corners=False)
    out = tensor[:, 0].numpy().transpose(1, 2, 0).astype(array.dtype)
    scale = source_side / side
    mapping = np.diag([scale, scale, 1., 1.])
    mapping[:2, 3] = (scale - 1) / 2
    return out, np.asarray(affine) @ mapping


def numpy_resize_image(array, side):
    """Independent bilinear reference for the exporter's PyTorch resize."""
    if array.shape[:2] == (side, side):
        return array.copy()
    coords = (np.arange(side, dtype=np.float64) + .5) * array.shape[0] / side - .5
    low = np.floor(coords).astype(int); weight = coords - low
    high = np.minimum(low + 1, array.shape[0] - 1); low = np.maximum(low, 0)
    xw, yw = weight[:, None, None], weight[None, :, None]
    out = (array[low[:, None], low[None, :], :] * (1-xw) * (1-yw)
           + array[low[:, None], high[None, :], :] * (1-xw) * yw
           + array[high[:, None], low[None, :], :] * xw * (1-yw)
           + array[high[:, None], high[None, :], :] * xw * yw)
    return out.astype(np.float32)


def independent_normalize(a):
    a = np.asarray(a, dtype=np.float32)
    positive = a[a > 0]
    if not positive.size:
        return np.zeros_like(a)
    low, high = map(float, np.percentile(positive, [1, 99]))
    if high <= low:
        return np.zeros_like(a)
    return np.clip((a-low)/(high-low), 0, 1).astype(np.float32)


def manual_crop(a, start, side=192):
    """NumPy verification path independent of crop_fixed."""
    start = np.asarray(start, dtype=int); end = start + side
    lo, hi = np.maximum(start, 0), np.minimum(end, a.shape[:2])
    offset = lo-start
    out = np.zeros((side, side, a.shape[2]), dtype=a.dtype)
    if np.all(hi > lo):
        out[offset[0]:offset[0]+hi[0]-lo[0], offset[1]:offset[1]+hi[1]-lo[1], :] = a[lo[0]:hi[0], lo[1]:hi[1], :]
    return out


def checksum_check(root):
    manifest = root / 'checksums.sha256'
    count, failed = 0, []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Invalid checksum path in {root}')
        count += 1
        if not path.is_file() or digest(path) != expected:
            failed.append(relative)
    if failed:
        raise ValueError(f'Checksum failures in {root}: {failed[:5]}')
    return {'files_verified': count, 'manifest_sha256': digest(manifest), 'failed': []}


def load_locked_inputs():
    protocol = json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    cases = json.loads((RUN/'cases.json').read_text(encoding='utf-8'))
    centers = json.loads((RUN/'predicted_centers.json').read_text(encoding='utf-8'))
    lock = json.loads((RUN/'prediction_lock.json').read_text(encoding='utf-8'))
    assert digest(RUN/'predicted_centers.json') == lock['centers_sha256']
    assert digest(RUN/'protocol.json') == lock['protocol_sha256']
    assert {k: len(v) for k, v in protocol['patients'].items()} == {'train':51,'val':13,'test_vol':16}
    all_ids = sum(protocol['patients'].values(), [])
    assert len(all_ids) == len(set(all_ids)) == len(cases) == 80
    assert set(cases) == set(centers) == set(all_ids)
    source_hashes, prediction_hashes = {}, {}
    for name, spec in protocol['models'].items():
        checkpoint = RUN/'models'/name/'best.pt'
        assert digest(checkpoint) == lock['models_sha256'][name]
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['membership']['train'] == spec['train']
    for case, meta in cases.items():
        pred = centers[case]; model_spec = protocol['models'][pred['model']]
        assert pred['training_patients'] == model_spec['train']
        assert case not in pred['training_patients']
        assert pred['model_sha256'] == lock['models_sha256'][pred['model']]
        assert case in protocol['patients'][meta['split']]
        if meta['split'] == 'train':
            assert case in model_spec['held_out_train']
        else:
            assert pred['model'] == 'final'
        for m, path in meta['paths'].items():
            value = digest(path)
            assert value == meta['source_sha256'][m], f'Source changed: {path}'
            source_hashes[path] = value
        folder = RUN/'predictions'/meta['center']/case
        pred_image = nib.load(folder/f'{case}_predicted_myocardium.nii.gz')
        mask = np.asarray(pred_image.dataobj)
        assert mask.shape == tuple(meta['grid_1mm_shape'])
        assert np.allclose(pred_image.affine, meta['grid_1mm_affine'], atol=1e-4)
        center, _, _ = predicted_center(mask > 0)
        assert center.tolist() == pred['center_hw_1mm']
        assert json.loads((folder/'localization.json').read_text(encoding='utf-8')) == pred
        for path in (folder/f'{case}_predicted_myocardium.nii.gz', folder/'localization.json'):
            prediction_hashes[str(path)] = digest(path)
    return protocol, cases, centers, lock, source_hashes, prediction_hashes


def export(protocol, cases, centers, lock):
    for root in DESTINATIONS.values():
        if root.exists():
            raise FileExistsError(f'Refusing to overwrite dataset: {root}')
    producer = Path(__file__).resolve()
    base_producer = WORKSPACE/'preprocessing'/'myopspp_unet2d.py'
    export_protocol = {
        'schema_version':1, 'date':'2026-10-27', 'crop_mm':192,
        'output_sides':[128,192], 'native_Z_retained':True,
        'center_source':str(RUN/'predicted_centers.json'),
        'center_sha256':lock['centers_sha256'],
        'center_policy':'reuse frozen predictions; no GT-guided changes',
        'image_resize':'bilinear, align_corners=False; identity at 192',
        'label_resize':'nearest-exact; identity at 192',
        'normalization':'positive-voxel p1/p99, per modality and output volume, after resize',
        'clipping_policy':'fixed 192 mm, keep all patients and slices including clipped/empty GT',
        'producer_sha256':digest(producer), 'base_producer_sha256':digest(base_producer),
        'dataset_generation_uses_no_new_training':True,
    }
    for pixels, root in DESTINATIONS.items():
        root.mkdir(parents=True)
        write_json(root/'status.json', {'status':'building', 'crop_mm':192, 'output_hw':[pixels,pixels]})
        write_json(root/'splits.json', {'dataset_id':root.name, 'patients':protocol['patients'], 'source_split_sha256':protocol['split_sha256']})
        write_json(root/'export_protocol.json', dict(export_protocol, output_hw=[pixels,pixels], spacing_xy_mm=192/pixels))
    records = {pixels:[] for pixels in DESTINATIONS}
    for case, info in sorted(cases.items()):
        pred = centers[case]
        shape, grid_affine = tuple(info['grid_1mm_shape']), np.asarray(info['grid_1mm_affine'])
        sources = {m:nib.load(path) for m,path in info['paths'].items()}
        expected_shape, expected_grid = target_grid(sources['C0'])
        assert expected_shape == shape and np.allclose(expected_grid, grid_affine, atol=1e-6)
        for m,n in sources.items():
            assert n.shape == sources['C0'].shape and np.allclose(n.affine, sources['C0'].affine, atol=1e-4)
            assert n.header.get_xyzt_units()[0] == 'mm'
        native_gt = np.asarray(sources['gd'].dataobj)
        gt = resample_xy(sources['gd'], shape, grid_affine, 0)
        cropped_gt, crop_affine, bounds = crop_fixed(gt, grid_affine, pred['center_hw_1mm'], 192)
        before, after_crop = label_counts(gt), label_counts(cropped_gt)
        loss = {k:before[k]-after_crop[k] for k in before}
        assert all(n >= 0 for n in loss.values())
        native_empty = [z for z in range(native_gt.shape[2]) if not np.any(native_gt[:,:,z])]
        cropped_images = {}
        for m in MODALITIES:
            grid = resample_xy(sources[m], shape, grid_affine, 1)
            arr, aff, image_bounds = crop_fixed(grid, grid_affine, pred['center_hw_1mm'], 192)
            assert np.allclose(aff, crop_affine) and image_bounds == bounds
            cropped_images[m] = arr
        for pixels, root in DESTINATIONS.items():
            folder = root/'cases'/info['center']/case; folder.mkdir(parents=True)
            target, output_affine = resize_to(cropped_gt, crop_affine, pixels, True)
            save_nii(folder/f'{case}_gd.nii.gz', target, output_affine, True)
            norms = {}
            for m, arr in cropped_images.items():
                output, image_affine = resize_to(arr, crop_affine, pixels)
                assert np.allclose(image_affine, output_affine)
                output, norms[m] = normalize(output)
                save_nii(folder/f'{case}_{m}.nii.gz', output, output_affine)
            native = root/'reference_native'/info['center']/case/f'{case}_gd.nii.gz'
            native.parent.mkdir(parents=True); shutil.copy2(info['paths']['gd'], native)
            loc_folder = root/'localization'/info['center']/case; loc_folder.mkdir(parents=True)
            for filename in (f'{case}_predicted_myocardium.nii.gz','localization.json'):
                shutil.copy2(RUN/'predictions'/info['center']/case/filename, loc_folder/filename)
            empty = [z for z in range(target.shape[2]) if not np.any(target[:,:,z])]
            meta = dict(info, case=case, localization='predicted_myocardium_unet2d_oof_train_final_val_test',
                        localizer=pred, prediction_lock_sha256=digest(RUN/'prediction_lock.json'),
                        crop_mm=192, crop=bounds, crop_affine=crop_affine.tolist(),
                        output_shape=list(target.shape), output_affine=output_affine.tolist(),
                        output_spacing_mm=nib.affines.voxel_sizes(output_affine).tolist(),
                        normalization=norms, normalization_method=export_protocol['normalization'],
                        labels='raw_0_200_500_600_1220_2221', native_label_counts=label_counts(native_gt),
                        counts_before_crop_1mm=before, counts_after_crop_before_resize=after_crop,
                        crop_loss_voxels_1mm=loss, counts_after_resize=label_counts(target),
                        classes_disappeared_during_resampling=[k for k,n in label_counts(native_gt).items() if n and not before[k]],
                        classes_disappeared_due_to_resize=[k for k,n in after_crop.items() if n and not np.any(target==int(k))],
                        source_empty_gt_z=native_empty, output_empty_gt_z=empty,
                        clipping_policy=export_protocol['clipping_policy'])
            write_json(folder/'metadata.json', meta)
            records[pixels].append(dict(case=case,center=info['center'],split=info['split'],
                                        localizer_model=pred['model'],empty_prediction=pred['empty_prediction'],
                                        before=before,after_crop=after_crop,crop_loss_voxels=loss))
        print(f'Exported {case}: both 192-mm variants', flush=True)
    environment = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                       scipy=scipy.__version__, nibabel=nib.__version__, torch=torch.__version__,
                       resampling_device='CPU', localizer_retrained=False)
    for pixels, root in DESTINATIONS.items():
        repro = root/'reproducibility'; repro.mkdir()
        for path in (producer, base_producer):shutil.copy2(path, repro/path.name)
        for name in ('protocol.json','environment.json','cases.json','predicted_centers.json','prediction_lock.json'):
            shutil.copy2(RUN/name, repro/name)
        write_json(repro/'export_environment.json', environment)
        for name in protocol['models']:
            folder = repro/'models'/name; folder.mkdir(parents=True)
            for filename in ('best.pt','complete.json','history.jsonl','best_validation.json'):
                shutil.copy2(RUN/'models'/name/filename, folder/filename)
        manifest = dict(dataset_id=root.name,patient_count=80,patients=sorted(cases),crop_mm=192,
                        model_hw=[pixels,pixels],output_inplane_spacing_mm=192/pixels,
                        localization='predicted U-Net2D; train 3-fold OOF; val/test final train-only model',
                        source=protocol['source'],shared_localizer_run=str(RUN),excluded_cases=[],
                        geometry=f'XY resample1mm; fixed 192mm crop; output{pixels}; native Z retained',
                        threshold=protocol['threshold'],labels=list(RAW_CODES),cases=records[pixels],
                        protocol_sha256=digest(RUN/'protocol.json'),prediction_lock=lock,
                        normalization=export_protocol['normalization'],
                        evaluation='Restore probabilities/logits to native grid before argmax for end-to-end evaluation; outside ROI background.',
                        training_cache_ready=False)
        write_json(root/'dataset_manifest.json', manifest)
        write_json(root/'status.json', {'status':'exported_pending_verification','crop_mm':192,'output_hw':[pixels,pixels]})


def independently_verify(cases, centers, lock):
    details = {pixels:[] for pixels in DESTINATIONS}
    file_checks = {pixels:[] for pixels in DESTINATIONS}
    empty_records = {pixels:[] for pixels in DESTINATIONS}
    disappeared = {pixels:[] for pixels in DESTINATIONS}
    for case, info in sorted(cases.items()):
        ref = nib.load(info['paths']['C0'])
        grid_shape, grid_affine = target_grid(ref)
        start = np.asarray(centers[case]['center_hw_1mm'])-96
        crop_mapping = np.eye(4); crop_mapping[:2,3] = start
        crop_affine = grid_affine@crop_mapping
        source_gt_image = nib.load(info['paths']['gd'])
        raw_gt = np.asarray(source_gt_image.dataobj)
        # Independent resampling/crop implementation and NumPy nearest indexing.
        full_gt = np.empty(grid_shape, dtype=np.uint16)
        mp = np.linalg.solve(source_gt_image.affine,grid_affine); mp[np.abs(mp)<1e-10]=0
        for z in range(grid_shape[2]):
            full_gt[:,:,z]=ndi.affine_transform(raw_gt[:,:,z],mp[:2,:2],mp[:2,3]+mp[:2,2]*z,
                                               output_shape=grid_shape[:2],output=np.uint16,order=0,
                                               mode='constant',cval=0,prefilter=False)
        crop_gt = manual_crop(full_gt,start)
        before, after = label_counts(full_gt), label_counts(crop_gt)
        loss = {k:before[k]-after[k] for k in before}
        reference_images = {}
        for m in MODALITIES:
            n = nib.load(info['paths'][m]); src = np.asarray(n.dataobj)
            mapping = np.linalg.solve(n.affine,grid_affine); mapping[np.abs(mapping)<1e-10]=0
            full = np.empty(grid_shape,dtype=np.float32)
            for z in range(grid_shape[2]):
                full[:,:,z]=ndi.affine_transform(src[:,:,z],mapping[:2,:2],mapping[:2,3]+mapping[:2,2]*z,
                                                output_shape=grid_shape[:2],output=np.float32,order=1,
                                                mode='constant',cval=0,prefilter=False)
            reference_images[m] = manual_crop(full,start)
        for pixels, root in DESTINATIONS.items():
            folder = root/'cases'/info['center']/case
            meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            scale = 192/pixels
            mapping = np.diag([scale,scale,1.,1.]); mapping[:2,3] = (scale-1)/2
            expected_affine = crop_affine@mapping
            indices = np.floor((np.arange(pixels)+.5)*192/pixels).astype(int)
            expected_gt = crop_gt[indices[:,None],indices[None,:],:]
            for m in (*MODALITIES,'gd'):
                path = folder/f'{case}_{m}.nii.gz'
                n = nib.load(path); a = np.asarray(n.dataobj)
                assert a.shape==(pixels,pixels,ref.shape[2])
                assert np.allclose(n.affine,expected_affine,atol=1e-4,rtol=1e-6)
                q, qc = n.get_qform(coded=True); s, sc = n.get_sform(coded=True)
                assert qc>0 and sc>0 and np.allclose(q,s,atol=1e-4)
                assert n.header.get_xyzt_units()[0]=='mm'
                assert np.allclose(nib.affines.voxel_sizes(n.affine),[scale,scale,info['native_spacing_mm'][2]],atol=1e-4)
                assert np.isfinite(a).all()
                if m=='gd':
                    assert a.dtype==np.uint16 and set(np.unique(a)).issubset(RAW_CODES)
                    assert np.array_equal(a,expected_gt),f'GT mismatch: {case}/{pixels}'
                    difference = 0.
                else:
                    expected_image = independent_normalize(numpy_resize_image(reference_images[m],pixels))
                    difference = float(np.max(np.abs(a-expected_image)))
                    assert a.dtype==np.float32 and a.min()>=0 and a.max()<=1
                    assert difference<=2e-6,f'Image reconstruction mismatch: {case}/{pixels}/{m}: {difference}'
                file_checks[pixels].append(dict(case=case,modality=m,path=str(path),shape='x'.join(map(str,a.shape)),
                                                sha256=digest(path),independent_reconstruction_max_abs_difference=difference,
                                                affine_and_units_verified=True))
            assert meta['localizer']==centers[case] and meta['crop_loss_voxels_1mm']==loss
            assert meta['counts_before_crop_1mm']==before and meta['counts_after_crop_before_resize']==after
            assert meta['counts_after_resize']==label_counts(expected_gt)
            native_copy = root/'reference_native'/info['center']/case/f'{case}_gd.nii.gz'
            assert digest(native_copy)==info['source_sha256']['gd']
            loc_folder = root/'localization'/info['center']/case
            for filename in (f'{case}_predicted_myocardium.nii.gz','localization.json'):
                assert digest(loc_folder/filename)==digest(RUN/'predictions'/info['center']/case/filename)
            row = dict(case=case,center=info['center'],split=info['split'],crop_mm=192,output_side=pixels,
                       spacing_xy_mm=scale,localizer_model=centers[case]['model'],
                       empty_prediction=centers[case]['empty_prediction'],
                       center_h=centers[case]['center_hw_1mm'][0],center_w=centers[case]['center_hw_1mm'][1])
            for group,codes in GROUPS.items():
                n = sum(before[str(k)] for k in codes); count = sum(loss[str(k)] for k in codes)
                row[group+'_voxels_before']=n;row[group+'_lost_voxels']=count
                row[group+'_lost_percent']=100*count/n if n else 0.
            details[pixels].append(row)
            for z in range(expected_gt.shape[2]):
                native_empty = not bool(np.any(raw_gt[:,:,z])); output_empty = not bool(np.any(expected_gt[:,:,z]))
                assert native_empty==output_empty
                if output_empty:
                    empty_records[pixels].append(dict(case=case,split=info['split'],z=z,slice_number_1based=z+1,
                                                      native_foreground_voxels=0,output_foreground_voxels=0))
                for code in RAW_CODES[1:]:
                    if np.any(crop_gt[:,:,z]==code) and not np.any(expected_gt[:,:,z]==code):
                        disappeared[pixels].append(dict(case=case,z=z,label_code=code))
        print(f'Verified geometry, labels and all modalities: {case}',flush=True)
    reports={}
    for pixels,root in DESTINATIONS.items():
        for name in lock['models_sha256']:
            checkpoint = root/'reproducibility'/'models'/name/'best.pt'
            assert digest(checkpoint)==lock['models_sha256'][name]
        for name in ('protocol.json','cases.json','predicted_centers.json','prediction_lock.json'):
            assert digest(root/'reproducibility'/name)==digest(RUN/name)
        splits = json.loads((root/'splits.json').read_text(encoding='utf-8'))
        expected_splits = json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))['patients']
        assert splits['patients']==expected_splits
        split_summary={}
        for split in ('train','val','test_vol','all'):
            rs=[r for r in details[pixels] if split=='all' or r['split']==split]
            split_summary[split]=dict(n=len(rs),**{g+'_clipped':sum(r[g+'_lost_voxels']>0 for r in rs) for g in GROUPS})
        report=dict(status='verified',patient_count=80,checked_output_nifti=len(file_checks[pixels]),
                    verified_native_gt_copies=80,verified_localizer_prediction_copies=80,
                    model_membership_and_locked_predictions_verified=True,
                    crop_mm=192,output_hw=[pixels,pixels],spacing_xy_mm=192/pixels,native_Z_retained=True,
                    split_summary=split_summary,
                    affected_cases={g:[r['case'] for r in details[pixels] if r[g+'_lost_voxels']>0] for g in GROUPS},
                    max_independent_image_reconstruction_difference=max(r['independent_reconstruction_max_abs_difference'] for r in file_checks[pixels]),
                    gt_all_voxels_match_independent_numpy=True,
                    classes_disappeared_due_to_resize_by_slice=disappeared[pixels],
                    source_empty_gt_slices=empty_records[pixels],
                    clipping_counts_measured_before_resize_on_1mm_grid=True,
                    limitation='Exact source reconstruction does not certify anatomical label correctness or preservation of every tiny lesion detail.')
        csv_write(root/'coverage_per_case.csv',details[pixels])
        csv_write(root/'file_verification.csv',file_checks[pixels])
        csv_write(root/'empty_gt_slices.csv',empty_records[pixels])
        write_json(root/'verification_report.json',report)
        reports[pixels]=report
    return reports


def visual_qa(cases):
    from PIL import Image,ImageDraw,ImageFont
    def font(size):return ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',size)
    def panel(a,g):
        lo,hi=np.percentile(a,[1,99]);a=np.clip((a-lo)/max(hi-lo,1e-6),0,1)
        rgb=np.repeat((a.T[::-1]*255).astype(np.uint8)[:,:,None],3,axis=2)
        for codes,color in (((200,1220,2221),(50,255,90)),((500,),(70,190,255)),((600,),(255,170,45))):
            mask=np.isin(g,codes);edge=mask&~ndi.binary_erosion(mask);rgb[edge.T[::-1]]=color
        return Image.fromarray(rgb).resize((192,192),Image.Resampling.NEAREST)
    selected=[('Case2001',3),('Case2017',0),('Case2018',0),('Case2031',1),
              ('Case2002',0),('Case2025',5),('Case3001',2),('Case3045',1)]
    qa=REPORT/'qa';qa.mkdir(parents=True,exist_ok=True)
    index=[]
    for start in range(0,len(selected),2):
        page=start//2+1
        canvas=Image.new('RGB',(716,958),'white');d=ImageDraw.Draw(canvas)
        d.text((7,5),'Actual exported NIfTI | green myocardium, blue LV, orange RV',font=font(18),fill='black')
        y=35
        for case,z in selected[start:start+2]:
            d.rectangle((0,y,716,y+28),fill='#e6edf5')
            d.text((7,y+3),f'{case} | z={z} (zero-based) | fixed FOV 192 mm',font=font(18),fill='black')
            for r,(pixels,root) in enumerate(DESTINATIONS.items()):
                yy=y+34+r*212
                folder=root/'cases'/cases[case]['center']/case
                gt=np.asarray(nib.load(folder/f'{case}_gd.nii.gz').dataobj)
                d.text((5,yy+76),f'{pixels}px',font=font(18),fill='black')
                d.text((5,yy+100),f'{192/pixels:g}mm',font=font(16),fill='black')
                for col,m in enumerate(MODALITIES):
                    x=115+col*198
                    d.text((x,yy),m,font=font(17),fill='black')
                    image=np.asarray(nib.load(folder/f'{case}_{m}.nii.gz').dataobj)
                    canvas.paste(panel(image[:,:,z],gt[:,:,z]),(x,yy+20))
            index.append(dict(case=case,z=z,page=page,filename=f'qa/qa_{page:02d}.png'))
            y+=460
        canvas.save(qa/f'qa_{page:02d}.png')
    write_json(REPORT/'qa_index.json',index)


def make_reports(reports):
    rows=[]
    for pixels,root in DESTINATIONS.items():
        r=reports[pixels];scale=192/pixels
        text=f'''# MyoPS++ U-Net2D ROI: 192 mm -> {pixels} pixels

Dataset hoàn tất ngày 2026-10-27. 80 ca (CenterB 35 + CenterC 45), split 51 train / 13 val / 16 test, giữ 359 lát theo một chuỗi.

- FOV XY: 192 x 192 mm; đầu ra {pixels} x {pixels} x D; spacing XY {scale:g} x {scale:g} mm. Giữ số lát, spacing và tọa độ Z native.
- Dùng lại tâm U-Net đã khóa: train dùng 3-fold OOF, val/test dùng final model chỉ học trên 51 train. Không huấn luyện lại, không dùng GT để sửa tâm hoặc bỏ ca.
- Cùng grid/crop cho C0, LGE, T2 và GT. Ảnh bilinear, nhãn nearest-exact; biến đổi affine theo pixel-center. Tại 192 pixel không resize ROI.
- Ảnh float32 [0,1], p1/p99 trên voxel dương, riêng từng modality và toàn volume đầu ra sau resize. GT uint16 mã gốc 0/200/500/600/1220/2221.
- Mapping khi đưa vào SCAR: 0/500/600 -> background; 200 -> normal myocardium; 1220 -> edema; 2221 -> scar.

## Kết quả kiểm chứng

- Đọc lại 320 NIfTI ảnh/GT; shape, affine, qform/sform, đơn vị mm, cường độ và mã nhãn đạt.
- Cả 80 GT khớp từng voxel với dựng lại độc lập phần crop/nearest bằng NumPy. 240 ảnh đối chiếu resize bằng bilinear NumPy; sai khác chuẩn hóa lớn nhất {r['max_independent_image_reconstruction_difference']:.9g}, trong ngưỡng 2e-6.
- Không cắt phạm nhãn cơ tim, scar/edema hoặc LV ở 80/80 ca trên grid 1 mm trước resize.
- Một ca cắt RV: Case2017 (train), 155/14471 voxel = 1.0711% trên grid 1 mm, trước resize. Giữ ca này, không sửa tâm/GT.
- Không có mã nhãn nào biến mất hoàn toàn trên từng slice do resize trong lần xuất này. Điều đó không chứng minh giữ nguyên mọi chi tiết nhỏ hoặc độ chính xác phân đoạn.
- Giữ nguyên 3 lát GT nguồn trống: Case2002 z=0,1 (train), Case2025 z=5 (test). Không tự tạo nhãn hay bỏ lát.
- 80 GT native và 80 mask định vị được sao chép nguyên bytes; checkpoint, membership và nguồn tâm được kiểm tra.

## Cấu trúc và sử dụng

- `cases/CenterB|CenterC/CaseXXXX/`: C0, LGE, T2, GT và metadata.
- `reference_native/`: GT nguồn nguyên bytes để đánh giá sau khôi phục.
- `localization/`: mask dự đoán trên toàn grid 1 mm và provenance tâm.
- `reproducibility/`: code, protocol, môi trường, fold membership, checkpoint và lịch sử huấn luyện localizer.
- `coverage_per_case.csv`, `file_verification.csv`, `empty_gt_slices.csv`, `verification_report.json`, `checksums.sha256`.

Đây là NIfTI export, chưa phải NPZ/H5 cache/profile training SCAR. Không gán thành các profile GT/oracle đang có. Đánh giá end-to-end cần khôi phục probabilities/logits về grid native trước argmax và dùng spacing native; ngoài ROI là background. Nếu chấm trực tiếp ROI thì công bố rõ grid đánh giá.

Các kiểm tra GT trên toàn cohort là audit mô tả; không tuyên bố test chưa từng được xem trong phát triển. Kết quả coverage chưa chứng minh phương án này cho Dice tốt hơn các bản khác. Chi tiết nhỏ có thể đổi qua nội suy/resample, đặc biệt bản 1.5 mm.
'''
        (root/'README.md').write_text(text,encoding='utf-8')
        rows.append(f'| {pixels} x {pixels} | {scale:g} x {scale:g} mm | 80 | 0/80 | 1/80 | `{root}` |')
    report='''# Báo cáo xuất hai dataset MyoPS++ FOV 192 mm

Ngày 2026-10-27. Dùng lại localizer U-Net2D và tâm dự đoán đã khóa, không train lại. Giữ 80/80 ca, split 51/13/16, 359 lát, toàn bộ Z native.

| Ma trận đầu ra | Spacing XY | Ca | Phạm cơ tim/bệnh lý/LV | Phạm RV | Thư mục |
|---|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

Coverage đo trên grid 1 mm trước resize: Case2017 (train) mất 155/14471 voxel RV = 1.0711%; ca này được giữ nguyên. Không tự tăng FOV hoặc sửa tâm theo GT.

Giữ nguyên các GT nguồn trống ở Case2002 z0/1 và Case2025 z5. Không có mã nhãn nào biến mất hoàn toàn ở cấp slice trong resize; không từ đó suy ra mọi chi tiết tổn thương đều được bảo toàn.

Hai bản gồm tổng cộng 640 NIfTI ảnh/GT chính, cùng 160 bản sao GT native, 160 mask định vị và đầy đủ provenance/checkpoint. Kiểm chứng từng voxel GT bằng crop/nearest NumPy; cường độ bằng bilinear NumPy. Đối chiếu hash nguồn, tâm và các bản export cũ trước/sau; kết quả hoàn tất ở `completion.json`.

Đây là NIfTI dataset, chưa tạo cache NPZ/H5, chưa tích hợp profile mới, chưa train model bệnh lý. Ảnh QA ở `qa/` được dựng từ các NIfTI đầu ra thực tế.
'''
    (REPORT/'BAO_CAO.md').write_text(report,encoding='utf-8')


def run():
    started=time.time();torch.set_num_threads(4)
    for root in DESTINATIONS.values():
        if root.exists():raise FileExistsError(f'Refusing to overwrite {root}')
    REPORT.mkdir(parents=True,exist_ok=True)
    print('Preflight: source hashes, old datasets, frozen centers, checkpoint membership...',flush=True)
    old_before={str(root):checksum_check(root) for root in OLD_EXPORTS}
    protocol,cases,centers,lock,source_hashes,prediction_hashes=load_locked_inputs()
    run_paths=[RUN/name for name in ('protocol.json','cases.json','predicted_centers.json','prediction_lock.json')]
    run_hashes={str(p):digest(p) for p in run_paths}
    write_json(REPORT/'preflight.json',dict(source_files=len(source_hashes),prediction_files=len(prediction_hashes),
                                          run_hashes=run_hashes,old_exports=old_before,
                                          destinations={str(k):str(v) for k,v in DESTINATIONS.items()}))
    export(protocol,cases,centers,lock)
    reports=independently_verify(cases,centers,lock)
    visual_qa(cases);make_reports(reports)
    for path,expected in {**source_hashes,**prediction_hashes,**run_hashes}.items():
        assert digest(path)==expected,f'Input changed during export: {path}'
    old_after={str(root):checksum_check(root) for root in OLD_EXPORTS}
    assert old_before==old_after
    integrity={}
    for pixels,root in DESTINATIONS.items():
        write_json(root/'status.json',dict(status='verified',crop_mm=192,output_hw=[pixels,pixels],
                                         spacing_xy_mm=192/pixels,patient_count=80))
        shutil.copy2(REPORT/'BAO_CAO.md',root/'BAO_CAO.md')
        with (root/'checksums.sha256').open('w',encoding='utf-8') as f:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.name!='checksums.sha256':
                    f.write(f'{digest(p)}  {p.relative_to(root).as_posix()}\n')
        integrity[str(root)]=checksum_check(root)
    completion=dict(status='complete',date='2026-10-27',destinations={str(k):str(v) for k,v in DESTINATIONS.items()},
                    main_nifti_verified=640,patients_per_dataset=80,slices_per_modality=359,
                    source_files_unchanged=len(source_hashes),localization_files_unchanged=len(prediction_hashes),
                    run_locks_unchanged=True,old_datasets_unchanged=old_after,new_dataset_integrity=integrity,
                    verified_reports={str(k):v for k,v in reports.items()},
                    elapsed_seconds=round(time.time()-started,2),visual_QA_pending_human_or_assistant_inspection=True)
    write_json(REPORT/'completion.json',completion)
    print(json.dumps(dict(status='complete',main_nifti=640,elapsed_seconds=completion['elapsed_seconds'],
                          datasets={str(k):str(v) for k,v in DESTINATIONS.items()},integrity=integrity),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    run()
