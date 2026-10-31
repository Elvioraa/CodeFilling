"""Fail-closed stage ownership, complete checkpoints, and epoch-boundary resume."""
import random
from pathlib import Path
import numpy as np
import torch
from opencood.utils.codefilling_protocol import MEMBERS, fingerprint, digest

SCHEMA = 'codefilling_doma_v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], generator=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA resume requires matching CUDA runtime')
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['generator'])


def read_checkpoint(path, dataset, stage):
    if not Path(path).is_file():
        raise FileNotFoundError('required parent checkpoint: '+str(path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('schema') != SCHEMA or checkpoint.get('dataset') != dataset or checkpoint.get('stage') != stage:
        raise ValueError('checkpoint schema/dataset/stage mismatch: '+str(path))
    if not checkpoint.get('validated', False):
        raise ValueError('parent is not a validation-selected checkpoint')
    cfg = checkpoint.get('config')
    if not cfg or checkpoint.get('config_hash') != fingerprint(cfg) or cfg['model']['core_method'] != 'codefilling_doma':
        raise ValueError('parent lacks a matching CodeFilling experiment config fingerprint')
    if cfg['fusion']['dataset'] != dataset or cfg['codefilling_protocol']['stage'] != stage:
        raise ValueError('parent config contradicts checkpoint metadata')
    return checkpoint


def initialize_parents(model, cfg, parents):
    protocol = cfg['codefilling_protocol']
    expected = protocol['parents']
    if set(parents) != set(expected):
        raise ValueError('required parent stages: '+str(expected))
    if not expected:
        return {}
    state = model.state_dict()
    provenance = {}
    if protocol['stage'] == 'detector':
        ownership = {}
        for m in MEMBERS:
            stage = 'pretrain_'+m
            checkpoint = read_checkpoint(parents[stage], protocol['dataset'], stage)
            origin = checkpoint['model']
            for key in state:
                member = next((x for x in MEMBERS if key.startswith(('encoder_'+x+'.', 'backbone_'+x+'.', 'aligner_'+x+'.'))), None)
                owner = 'pretrain_'+member if member else 'pretrain_m1'
                # New shared codebook is deliberately initialized here, then fitted.
                if key.startswith('transport.quantizer.'):
                    continue
                if owner != stage:
                    continue
                if key not in origin or origin[key].shape != state[key].shape:
                    raise ValueError('missing/incompatible parent tensor: '+key)
                state[key] = origin[key]
                ownership[key] = stage
            provenance[stage] = dict(path=str(parents[stage]), sha256=digest(parents[stage]))
        required = {k for k in state if not k.startswith('transport.quantizer.')}
        if set(ownership) != required:
            raise ValueError('incomplete tensor ownership')
        model.load_state_dict(state, strict=True)
    else:
        stage = expected[0]
        checkpoint = read_checkpoint(parents[stage], protocol['dataset'], stage)
        model.load_state_dict(checkpoint['model'], strict=True)
        provenance[stage] = dict(path=str(parents[stage]), sha256=digest(parents[stage]))
    return provenance


def save_checkpoint(path, model, optimizer, scheduler, cfg, epoch, best, generator, parents, validated=False):
    path = Path(path)
    state = dict(schema=SCHEMA, dataset=cfg['fusion']['dataset'], stage=cfg['codefilling_protocol']['stage'],
                 config=cfg, config_hash=fingerprint(cfg), epoch=epoch, best=best, validated=validated,
                 model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 rng=rng_state(generator), parents=parents)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def resume_checkpoint(path, model, optimizer, scheduler, cfg, generator):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema') != SCHEMA or state.get('config_hash') != fingerprint(cfg):
        raise ValueError('resume config/schema mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'], generator)
    return state
