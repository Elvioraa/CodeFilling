"""DOMA protocol contracts. No sensor/CUDA imports, also used by the CPU gates."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

MEMBERS = ('m1', 'm2', 'm3', 'm4')
REQUIRED = 'SERVER_VALIDATION_REQUIRED'
ROOT = Path(__file__).resolve().parents[2]
V2XSET_EXCLUDED_SCENE = '2021_09_09_13_20_58'


def experiment_output(cfg, output=None):
    """One fixed dataset/seed root, one directory per existing stage. No mkdir."""
    dataset = {'opv2v': 'OPV2V', 'v2xset': 'V2XSET'}[cfg['fusion']['dataset']]
    if cfg['seed'] != 303:
        raise ValueError('this formal output protocol is SEED303')
    expected = ROOT/'opencood/logs'/('CodeFilling_'+dataset+'_SEED303')/cfg['codefilling_protocol']['stage']
    if output is not None and Path(output).resolve() != expected.resolve():
        raise ValueError('formal stage output must be '+str(expected))
    return expected


def validate_input_members(ids, selected, mods, assigned_mods, requested):
    """Validate the physical prefix AFTER range filtering, allowing modality gaps."""
    if (requested not in (1,2,3,4) or not ids or len(ids) != len(set(ids)) or
            len(assigned_mods) != len(ids) or any(m not in MEMBERS for m in assigned_mods)):
        raise ValueError('invalid evaluation CAV IDs/count/modalities')
    realized = min(requested, len(ids))
    if selected != ids[:realized] or mods != assigned_mods[:realized] or mods[0] != 'm1':
        raise ValueError('invalid realized physical/member prefix or CAV/modality mapping')
    return realized


def validate_evaluation_frame(row):
    """Audit real CAV/modality correspondence and sample/GT identity across groups."""
    count = row['requested_cavs']
    ids, selected, mods = row['all_cav_ids'], row['input_cav_ids'], row['modalities']
    assigned_mods = row['cav_modalities']
    realized = validate_input_members(ids, selected, mods, assigned_mods, count)
    if row['realized_cavs'] != realized or selected != ids[:realized]:
        raise ValueError('invalid realized physical/member prefix')
    ledger = row['communication']
    expected = list(range(1, realized))
    if (ledger['peers'] != len(expected) or
            [message['sender'] for message in ledger['messages']] != expected or
            any(message['batch'] != 0 for message in ledger['messages'])):
        raise ValueError('communication sender set differs from actual input prefix')
    for message in ledger['messages']:
        index = message['sender']
        message['cav_id'], message['modality'] = selected[index], mods[index]
    key = row['sample_key']
    if not key.get('scene') or not key.get('timestamp') or key.get('sample_idx') != row['sample_idx']:
        raise ValueError('missing/mismatched actual scene/timestamp identity')
    # Index alone cannot establish that two DataLoaders used the same frame.
    return fingerprint(dict(sample_key=key, all_cav_ids=ids, cav_modalities=assigned_mods, gt_hash=row['gt_hash']))


def evaluation_sample_key(dataset, index):
    """Resolve the native base dataset's index to its actual scene/timestamp."""
    if index < 0 or not dataset.len_record or index >= dataset.len_record[-1]:
        raise ValueError('invalid evaluation sample index')
    scenario = next(i for i, end in enumerate(dataset.len_record) if index < end)
    local = index - (dataset.len_record[scenario-1] if scenario else 0)
    timestamp = dataset.return_timestamp_key(dataset.scenario_database[scenario], local+dataset.tau)
    return dict(scene=Path(dataset.scenario_folders[scenario]).name, timestamp=str(timestamp), sample_idx=index)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config(cfg):
    p = cfg['codefilling_protocol']
    ds = cfg['fusion']['dataset']
    if ds not in ('opv2v', 'v2xset') or p['dataset'] != ds:
        raise ValueError('dataset mismatch')
    if cfg['heter']['ego_modality'] != p['ego_modality']:
        raise ValueError('ego mismatch')
    stages = ['pretrain_'+m for m in MEMBERS] + ['detector','codebook','finetune','infer']
    stage = p['stage']
    if stage not in stages or cfg['model']['core_method'] != 'codefilling_doma':
        raise ValueError('invalid CodeFilling stage/model')
    mods = [stage[-2:]] if stage.startswith('pretrain_') else list(MEMBERS)
    model_mods = [m for m in MEMBERS if m in cfg['model']['args']]
    if set(cfg['heter']['modality_setting']) != set(mods) or set(model_mods) != set(mods):
        raise ValueError('stage member set mismatch')
    dependencies = {'detector':['pretrain_'+m for m in MEMBERS], 'codebook':['detector'],
                    'finetune':['codebook'], 'infer':['finetune']}
    if p['parents'] != dependencies.get(stage, []):
        raise ValueError('stage parent DAG mismatch')
    expected = {'m1': 'point_pillar', 'm2': 'lift_splat_shoot', 'm3': 'second', 'm4': 'lift_splat_shoot'}
    for m, setting in cfg['heter']['modality_setting'].items():
        model = cfg['model']['args'][m]
        if setting['core_method'] != expected[m] or model['core_method'] != expected[m]:
            raise ValueError('invalid member encoder: ' + m)
        sensor = 'lidar' if m in ('m1', 'm3') else 'camera'
        if setting['sensor_type'] != sensor or model['sensor_type'] != sensor:
            raise ValueError('invalid sensor: ' + m)
        if sensor == 'camera':
            if model['encoder_args']['camera_encoder'] != {'m2': 'EfficientNet', 'm4': 'Resnet101'}[m]:
                raise ValueError('invalid camera trunk: ' + m)
        elif model['encoder_args']['voxel_size'] != setting['preprocess']['args']['voxel_size']:
            raise ValueError('voxel mismatch')
    channels = cfg['heter'].get('lidar_channels_dict', {})
    if ds == 'v2xset' and channels:
        raise ValueError('V2XSet must use original PCDs')
    if ds == 'opv2v' and 'm3' in cfg['heter']['modality_setting'] and channels != {'m3': 32}:
        raise ValueError('OPV2V m3 must use verified _32.pcd')
    cb = cfg['model']['args']['codebook']
    if cb['seg_num'] < 1 or 64 % cb['seg_num'] or not 1 <= cb['r'] <= 3 or cb['dict_size'] < 2:
        raise ValueError('invalid codebook geometry')
    if p['stage'] == 'infer' and (p['ego_modality'] != 'm1' or cfg['heter']['mapping_dict'] != dict(zip(MEMBERS, MEMBERS))):
        raise ValueError('evaluation must fix m1 ego and preserve all four modalities')
    # Training/inference must consume the same locked sensor evidence as the
    # standalone checker. Budget/optimizer sweeps can remain separate configs.
    lock = json.loads((ROOT/p['source_lock']).read_text(encoding='utf-8'))
    if lock['dataset'] != ds:
        raise ValueError('source lock dataset mismatch')
    for source in lock['sources']:
        if digest(ROOT/source['path']) != source['sha256']:
            raise ValueError('source evidence changed: '+source['path'])
    locked_assignment = {s['path'] for s in lock['sources'] if s['path'].endswith('.json')}
    if cfg['heter']['assignment_path'] not in locked_assignment:
        raise ValueError('assignment is not in the source lock')
    source_name = mods[0] if stage.startswith('pretrain_') else 'final'
    reference = load_config(ROOT/'docs/codefilling_sources'/ds/(source_name+'.yaml'))
    for m in mods:
        if (cfg['heter']['modality_setting'][m] != reference['heter']['modality_setting'][m] or
                cfg['model']['args'][m] != reference['model']['args'][m]):
            raise ValueError('member sensor contract differs from DOMA: '+m)
    return cfg


def ordered_cavs(cavs, assignment):
    """Match DOMA's sorted vehicle order, moving the negative roadside ID last."""
    cavs = sorted(cavs)
    if cavs and int(cavs[0]) < 0:
        cavs = cavs[1:] + cavs[:1]
    if set(cavs) != set(assignment):
        raise ValueError('assignment/data CAV set differs')
    mods = [assignment[c] for c in cavs]
    if mods != [MEMBERS[i % 4] for i in range(len(cavs))]:
        raise ValueError('ordered assignment is not m1/m2/m3/m4 cyclic')
    return cavs


def lidar_path(path, dataset, modality):
    """Rewrite a path component, never an arbitrary substring; never V2XSet."""
    path = str(path).replace('\\', '/')
    if dataset == 'v2xset' or modality != 'm3':
        return path
    parts = path.split('/')
    if 'OPV2V' not in parts:
        raise ValueError('OPV2V-H requires an OPV2V root component')
    parts[parts.index('OPV2V')] = 'OPV2V_Hetero'
    return '/'.join(parts)[:-4] + '_32.pcd'


def apply_server_settings(cfg, path):
    """Explicit receipt, not an inferred server protocol. Range is six metric bounds."""
    settings = load_config(path)
    if settings.get('status') != 'VERIFIED' or not settings.get('evidence'):
        raise ValueError(REQUIRED + ': receipt needs VERIFIED and evidence')
    if settings['dataset'] != cfg['fusion']['dataset']:
        raise ValueError('receipt dataset mismatch')
    reference = settings.get('doma_formal_evaluation', {})
    if (reference.get('status') != 'VERIFIED' or not reference.get('command') or
            not reference.get('evidence')):
        raise ValueError(REQUIRED + ': actual DOMA formal command/evidence must be verified')
    for key in ('eval_range', 'comm_range', 'test_dir'):
        if settings.get(key) != reference.get(key) or settings.get(key) is None:
            raise ValueError(REQUIRED + ': CodeFilling '+key+' differs from DOMA formal evaluation')
    if settings['dataset'] == 'v2xset':
        if (reference.get('excluded_scenes') != [V2XSET_EXCLUDED_SCENE] or
                not reference.get('scene_policy_evidence')):
            raise ValueError(REQUIRED + ': V2XSet formal excluded-scene policy is unverified or differs')
    bounds = settings['eval_range']
    if len(bounds) != 6 or any(bounds[i] >= bounds[i + 3] for i in range(3)) or settings['comm_range'] <= 0:
        raise ValueError('invalid range')
    if any(abs(bounds[i] + bounds[i + 3]) > 1e-6 for i in (0, 1)):
        raise ValueError('this adapter requires origin-centered XY grids')
    if any(abs((bounds[i+3]-bounds[i])/.8-round((bounds[i+3]-bounds[i])/.8)) > 1e-5 or
           round((bounds[i+3]-bounds[i])/.8) % 4 for i in (0,1)):
        raise ValueError('evaluation grid must fit three native scales at 0.8m')
    cfg = copy.deepcopy(cfg)
    cfg['comm_range'] = settings['comm_range']
    cfg['cav_lidar_range'] = bounds
    cfg['preprocess']['cav_lidar_range'] = bounds
    cfg['postprocess']['gt_range'] = bounds
    cfg['postprocess']['anchor_args']['cav_lidar_range'] = bounds
    cfg['model']['args']['lidar_range'] = bounds
    for m, setting in cfg['heter']['modality_setting'].items():
        model = cfg['model']['args'][m]
        if setting['sensor_type'] == 'lidar':
            setting['preprocess']['cav_lidar_range'] = bounds
            model['encoder_args']['lidar_range'] = bounds
        else:
            model['camera_mask_args']['cav_lidar_range'] = bounds
    cfg['validate_dir'] = settings['test_dir']
    cfg['test_dir'] = settings['test_dir']
    cfg['codefilling_protocol']['server_receipt'] = settings
    return cfg
