"""Extract immutable local DOMA evidence and create independent experiments.

This command is intentionally non-overwriting. Existing files must match exactly.
It never imports or loads DOMA model weights.
"""
import argparse
import ast
import copy
import json
from pathlib import Path

import yaml
from opencood.utils.codefilling_protocol import ROOT, MEMBERS, digest


def put(path, data):
    path = Path(path)
    data = data.encode('utf-8') if isinstance(data, str) else data
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError('refusing to overwrite: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def dump(value):
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def prepare(doma):
    doma = Path(doma)
    history = doma / 'opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2_R2_TABLE4_EXT/history_snapshot.json'
    snapshot = json.loads(history.read_text(encoding='utf-8'))
    prefix = 'opencood/logs/DOMA_V2_R2_2GPU_LR1P414_SEED303/'
    entries = {'m1': 'stage1/m1_base/config.yaml', 'm2': 'stage2/m2_alignto_m1_lr0p0012/config.yaml',
               'm3': 'stage2/m3_alignto_m1_lr0p0030/config.yaml', 'm4': 'stage2/m4_alignto_m1_lr0p0017/config.yaml',
               'final': 'final_infer/config.yaml'}
    for ds in ('opv2v', 'v2xset'):
        evidence = ROOT / 'docs/codefilling_sources' / ds
        references, sources = {}, []
        for m, suffix in entries.items():
            if ds == 'opv2v':
                content = snapshot[prefix + suffix].encode()
                origin = str(history) + '#' + prefix + suffix
            else:
                rel = 'final_infer/m1m2m3m4.yaml' if m == 'final' else ('stage1/m1.yaml' if m == 'm1' else 'stage2/'+m+'.yaml')
                source = doma / 'opencood/hypes_yaml/v2xset/MoreModality/DOMA/V2' / rel
                content = source.read_bytes()
                origin = str(source)
            target = evidence / (m + '.yaml')
            put(target, content)
            sources.append(dict(path=str(target.relative_to(ROOT)).replace('\\','/'), sha256=digest(target), origin=origin))
            references[m] = yaml.safe_load(content)
        for suffix in ('', '_in_order'):
            source = doma / ('opencood/modality_assign/'+ds+'_4modality'+suffix+'.json')
            target = ROOT / 'opencood/modality_assign' / source.name
            put(target, source.read_bytes())
            sources.append(dict(path=str(target.relative_to(ROOT)).replace('\\','/'), sha256=digest(target), origin=str(source)))
        lock = dict(dataset=ds, status='SERVER_VALIDATION_REQUIRED',
                    basis='OPV2V saved Full V2_R2 config; V2XSet local V2 config', sources=sources,
                    unresolved=['actual server split/data identity', 'eval_range', 'comm_range', 'GPU sensor forward/AP',
                                'CodeFilling training budget and codebook/solver sweep'],
                    adaptation_choices=dict(codebook={'seg_num': 2, 'dict_size': 64, 'r': 3}, fusion='native max',
                                            budget='local experiment template, not a DOMA/CodeFilling published result'))
        put(evidence/'lock.json', json.dumps(lock, indent=2, ensure_ascii=False)+'\n')
        exp = ROOT / ('opencood/hypes_yaml/'+ds+'/doma_codefilling')
        final = references['final']

        def experiment(stage, member=None):
            source = references[member] if member else final
            cfg = copy.deepcopy(source)
            cfg.pop('validation_detection', None)
            cfg.pop('stage2_added_modality', None)
            cfg['name'] = 'CodeFilling_DOMA_'+ds+'_'+stage
            cfg['seed'] = 303
            cfg['data_augment'] = cfg.get('data_augment', [])
            cfg['time_delay'] = 0
            cfg['train_params'] = dict(batch_size=1, epoches=30, max_cav=5, eval_freq=1, save_freq=1, single_weight=1.0)
            mods = [member] if member else list(MEMBERS)
            cfg['heter']['ego_modality'] = member or 'm1'
            cfg['heter']['mapping_dict'] = {m: member or m for m in MEMBERS}
            cfg['heter']['assignment_path'] = 'opencood/modality_assign/'+ds+'_4modality'+('_in_order' if stage == 'infer' else '')+'.json'
            cfg['heter']['modality_setting'] = {m: copy.deepcopy(source['heter']['modality_setting'][m]) for m in mods}
            cfg['heter']['lidar_channels_dict'] = {'m3':32} if ds == 'opv2v' and 'm3' in mods else {}
            args = copy.deepcopy(source['model']['args'])
            for key in list(args):
                if key == 'doma' or key == 'stage2_added_modality' or (key in MEMBERS and key not in mods):
                    args.pop(key)
            args.update(ego_modality=member or 'm1', supervise_single=True, in_head_single=64,
                        stage=stage, codebook=dict(seg_num=2, dict_size=64, r=3), fusion_method='max',
                        fusion_args=dict(communication=dict(thre=0.0, solver=dict(thre=1.0, method='sum'))))
            # Keep the encoder/backbone/aligner from each matching DOMA member.
            if not member:
                for m in MEMBERS:
                    args[m] = copy.deepcopy(final['model']['args'][m])
            cfg['model'] = dict(core_method='codefilling_doma', args=args)
            # CodeFilling has detection/depth supervision, no DOMA occupancy/object loss.
            cfg['loss'] = dict(core_method='point_pillar_depth_loss', args=copy.deepcopy(source['loss']['args']))
            cfg['loss']['args'].pop('pyramid', None)
            cfg['codefilling_protocol'] = dict(dataset=ds, stage=stage, ego_modality=member or 'm1',
                status='SERVER_VALIDATION_REQUIRED', source_lock=str((evidence/'lock.json').relative_to(ROOT)).replace('\\','/'),
                budget_source='adaptation_template_requires_server_validation',
                parents=[], checkpoint_schema='codefilling_doma_v1')
            if stage == 'infer':
                cfg['use_cav'] = 4
                cfg['codefilling_protocol']['parents'] = ['finetune']
            elif stage == 'detector':
                cfg['codefilling_protocol']['parents'] = ['pretrain_'+m for m in MEMBERS]
            elif stage == 'codebook':
                cfg['codefilling_protocol']['parents'] = ['detector']
            elif stage == 'finetune':
                cfg['codefilling_protocol']['parents'] = ['codebook']
            put(exp/(stage+'.yaml'), dump(cfg))
        for m in MEMBERS:
            experiment('pretrain_'+m, m)
        for stage in ('detector','codebook','finetune','infer'):
            experiment(stage)
    # Copy the exact DOMA GT-before-input-cutoff implementation with provenance.
    source = doma/'opencood/data_utils/datasets/heter_infer/intermediate_heter_infer_fusion_dataset.py'
    put(ROOT/'opencood/data_utils/datasets/codefilling_infer_dataset.py', source.read_bytes())
    # Native MaxFusion kept AST-identical while avoiding unrelated model-zoo imports.
    source = ROOT/'opencood/models/fuse_modules/fusion_in_one.py'
    text = source.read_text(encoding='utf-8')
    tree = ast.parse(text)
    parts = [ast.get_source_segment(text, n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ('regroup', 'MaxFusion')]
    put(ROOT/'opencood/models/fuse_modules/codefilling_max.py',
        '"""Native MaxFusion and regroup, extracted verbatim from fusion_in_one.py."""\nimport torch\nfrom torch import nn\nfrom opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple\n\n'+'\n\n'.join(parts)+'\n')
    source = doma/'opencood/models/sub_modules/lss_submodule.py'
    text = source.read_text(encoding='utf-8')
    tree = ast.parse(text)
    parts = [ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('CamEncode','CamEncode_Resnet101')]
    target = ROOT/'opencood/models/sub_modules/codefilling_camera.py'
    put(target, '"""Exact local DOMA camera contracts; CodeFilling codebook/filling stay native."""\n'
        'import torch\nfrom torch import nn\nimport torch.nn.functional as F\n'
        'from efficientnet_pytorch import EfficientNet\nfrom torchvision.models.resnet import resnet101\n'
        'from opencood.utils.camera_utils import bin_depths\nfrom opencood.models.sub_modules.lss_submodule import Up\n\n'+
        '\n\n'.join(parts)+'\n')
    put(ROOT/'docs/codefilling_sources/camera_source.json', json.dumps(dict(origin=str(source),
        source_sha256=digest(source), adapter_sha256=digest(target), classes=['CamEncode','CamEncode_Resnet101']),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doma-root', required=True)
    prepare(parser.parse_args().doma_root)
