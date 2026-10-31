"""Windows static protocol gate and optional exhaustive on-disk sensor audit."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
from opencood.utils.codefilling_protocol import ROOT, MEMBERS, digest, load_config, validate_config, ordered_cavs, lidar_path


def check_static(dataset):
    root = ROOT/('opencood/hypes_yaml/'+dataset+'/doma_codefilling')
    lock = json.loads((ROOT/'docs/codefilling_sources'/dataset/'lock.json').read_text(encoding='utf-8'))
    for source in lock['sources']:
        if digest(ROOT/source['path']) != source['sha256']:
            raise ValueError('source evidence hash mismatch: '+source['path'])
    refs = {m: load_config(ROOT/'docs/codefilling_sources'/dataset/(m+'.yaml')) for m in (*MEMBERS, 'final')}
    checked = []
    for path in sorted(root.glob('*.yaml')):
        cfg = validate_config(load_config(path))
        source = refs[cfg['codefilling_protocol']['stage'].replace('pretrain_', '')] if path.stem.startswith('pretrain_') else refs['final']
        for m in cfg['heter']['modality_setting']:
            if cfg['heter']['modality_setting'][m] != source['heter']['modality_setting'][m]:
                raise ValueError('sensor protocol differs: '+str(path)+' '+m)
            if cfg['model']['args'][m] != source['model']['args'][m]:
                raise ValueError('encoder/backbone/aligner differs: '+str(path)+' '+m)
        for key in ('cav_lidar_range', 'comm_range', 'postprocess', 'preprocess', 'input_source', 'label_type'):
            if cfg[key] != source[key]:
                raise ValueError('DOMA protocol field differs: '+key+' '+str(path))
        for parent in cfg['codefilling_protocol']['parents']:
            if not (root/(parent+'.yaml')).is_file():
                raise ValueError('parent config missing')
        checked.append(str(path.relative_to(ROOT)).replace('\\','/'))
    if len(checked) != 8:
        raise ValueError('expected four pretrain + detector + codebook + finetune + infer')
    camera = json.loads((ROOT/'docs/codefilling_sources/camera_source.json').read_text(encoding='utf-8'))
    if digest(ROOT/'opencood/models/sub_modules/codefilling_camera.py') != camera['adapter_sha256']:
        raise ValueError('camera source changed; redo source audit')
    assignment = json.loads((ROOT/'opencood/modality_assign'/(dataset+'_4modality_in_order.json')).read_text())
    for scene, cavs in assignment.items():
        ordered_cavs(list(cavs), cavs)
    return dict(status='PASS_STATIC', stages=checked, ordered_scenes=len(assignment),
                server_status='SERVER_VALIDATION_REQUIRED', unresolved=lock['unresolved'],
                pre_run_blockers=(['actual DOMA formal command/eval_range/comm_range/test_dir agreement'] +
                    (['formal V2XSet evaluation excludes 2021_09_09_13_20_58: local source only; server evidence missing']
                     if dataset == 'v2xset' else [])),
                training_budget='batch=1/epoch=30 are editable provisional values; formal budget requires review')


def pcd_header(path):
    fields = None
    with path.open('rb') as stream:
        for _ in range(100):
            line = stream.readline().decode('ascii', errors='replace').strip()
            if line.startswith('FIELDS '):
                fields = line.split()[1:]
            if line.startswith('DATA '):
                break
    if not fields or not {'x','y','z'}.issubset(fields) or not ({'rgb','rgba','intensity'} & set(fields)):
        raise ValueError('PCD lacks XYZI/RGB information: '+str(path))
    return fields


def audit_data(dataset, frames_per_scene=0):
    cfg = load_config(ROOT/('opencood/hypes_yaml/'+dataset+'/doma_codefilling/infer.yaml'))
    counts = dict(scenes=0, frames=0, pcds=0, cameras=0, depth=0, visibility=0)
    for split, key in (('train','root_dir'), ('validate','validate_dir'), ('test','test_dir')):
        root = ROOT/cfg[key]
        if not root.is_dir():
            raise FileNotFoundError('SERVER_VALIDATION_REQUIRED: '+str(root))
        assignment_file = dataset+'_4modality'+('_in_order' if split == 'test' else '')+'.json'
        assignment = json.loads((ROOT/'opencood/modality_assign'/assignment_file).read_text())
        for scene in sorted(p for p in root.iterdir() if p.is_dir()):
            if dataset == 'v2xset' and scene.name == '2021_09_09_13_20_58':
                continue
            if scene.name not in assignment:
                raise ValueError('scene missing from assignment: '+str(scene))
            cavs = sorted(p.name for p in scene.iterdir() if p.is_dir())
            if split == 'test':
                cavs = ordered_cavs(cavs, assignment[scene.name])
            elif set(cavs) != set(assignment[scene.name]):
                raise ValueError('training CAV assignment mismatch')
            timestamps = sorted(p.stem for p in (scene/cavs[0]).glob('*.yaml') if 'additional' not in p.name)
            timestamps = [t for t in timestamps if not (scene.name == '2021_08_20_21_10_24' and t == '000265')]
            if not timestamps:
                raise ValueError('empty scene: '+str(scene))
            if frames_per_scene:
                timestamps = timestamps[:frames_per_scene]
            counts['scenes'] += 1
            for stamp in timestamps:
                counts['frames'] += 1
                for cav in cavs[:cfg['train_params']['max_cav']]:
                    folder = scene/cav
                    meta = folder/(stamp+'.yaml')
                    if not meta.is_file():
                        raise FileNotFoundError(str(meta))
                    data = load_config(meta)
                    if 'lidar_pose' not in data or 'vehicles' not in data:
                        raise ValueError('missing pose/objects: '+str(meta))
                    # Training assignment may reassign any CAV to any member: audit
                    # all raw/32 sensors per CAV, not only its test slot.
                    modalities = ('m1','m3') if split != 'test' else (assignment[scene.name][cav],)
                    for m in set(('m1',)+modalities):
                        if m in ('m1','m3'):
                            pcd_header(Path(lidar_path(folder/(stamp+'.pcd'), dataset, m)))
                            counts['pcds'] += 1
                    hdf5 = folder/(stamp+'_imgs.hdf5')
                    if hdf5.is_file():
                        import h5py
                        with h5py.File(hdf5,'r') as bundle:
                            for i in range(4):
                                for kind in (('camera','depth') if dataset == 'opv2v' else ('camera',)):
                                    if kind+str(i) not in bundle:
                                        raise ValueError('HDF5 missing '+kind+str(i)+': '+str(hdf5))
                    for i in range(4):
                        if 'camera'+str(i) not in data or not {'cords','intrinsic'}.issubset(data['camera'+str(i)]):
                            raise ValueError('missing camera calibration: '+str(meta))
                        for kind in (('camera','depth') if dataset == 'opv2v' else ('camera',)):
                            sensor = folder/(stamp+'_'+kind+str(i)+'.png')
                            if kind == 'depth':
                                sensor = Path(str(sensor).replace('OPV2V','OPV2V_Hetero'))
                            if not hdf5.is_file() and not sensor.is_file():
                                raise FileNotFoundError(str(sensor))
                            counts['cameras' if kind == 'camera' else 'depth'] += 1
                    if dataset == 'opv2v':
                        vis = folder/(stamp+'_bev_visibility.png')
                        alternative = ROOT/'dataset/OPV2V/additional'/split/scene.name/cav/vis.name
                        if not vis.is_file() and not alternative.is_file():
                            raise FileNotFoundError(str(vis))
                        counts['visibility'] += 1
    return dict(status='PASS_SAMPLED' if frames_per_scene else 'PASS_FULL_FILE_AUDIT', counts=counts,
                limitation='file/header/calibration presence; actual sensor decode and CUDA forward still required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['opv2v','v2xset','both'], default='both')
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--frames-per-scene', type=int, default=0)
    parser.add_argument('--output')
    opt = parser.parse_args()
    if opt.frames_per_scene < 0:
        parser.error('--frames-per-scene must be nonnegative')
    result = dict(python=sys.version, dependencies={name:bool(importlib.util.find_spec(name)) for name in
        ('torch','yaml','numpy','pytest','icecream','spconv','timm','mmcv','torch_scatter','tensorboardX')})
    data_failed = False
    for ds in (('opv2v','v2xset') if opt.dataset == 'both' else (opt.dataset,)):
        result[ds] = check_static(ds)
        if opt.check_data:
            try:
                result[ds]['data'] = audit_data(ds, opt.frames_per_scene)
            except (FileNotFoundError, ValueError) as error:
                result[ds]['data'] = dict(status='SERVER_VALIDATION_REQUIRED', error=str(error))
                data_failed = True
    print(json.dumps(result, indent=2))
    if opt.output:
        Path(opt.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if data_failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
