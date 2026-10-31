"""Independent native-algorithm stage trainer. No automatic test-set invocation."""
import argparse
import json
import torch
import yaml
from opencood.utils.codefilling_protocol import load_config, validate_config, experiment_output
from opencood.utils.codefilling_state import seed_all, initialize_parents, save_checkpoint, resume_checkpoint


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


def stage_loss(model, batch, criterion, cfg):
    output = model(batch['ego'])
    if cfg['codefilling_protocol']['stage'] == 'codebook':
        return output['codebook_loss']
    return (criterion(output, batch['ego']['label_dict']) +
            criterion(output, batch['ego']['label_dict_single'], suffix='_single') *
            cfg['train_params'].get('single_weight', 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', help='optional explicit canonical stage path under opencood/logs/CodeFilling_*_SEED303/')
    parser.add_argument('--parent', action='append', default=[], help='stage=checkpoint.pth')
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check-only', action='store_true')
    opt = parser.parse_args()
    cfg = validate_config(load_config(opt.config))
    stage = cfg['codefilling_protocol']['stage']
    if stage == 'infer':
        raise ValueError('use inference_codefilling_doma for infer config')
    output = experiment_output(cfg, opt.output)
    parents = dict(value.split('=', 1) for value in opt.parent)
    if not opt.resume and set(parents) != set(cfg['codefilling_protocol']['parents']):
        raise ValueError('missing or unexpected --parent; see config parents')
    if opt.check_only:
        from opencood.utils.codefilling_state import read_checkpoint
        for name, path in parents.items():
            read_checkpoint(path, cfg['fusion']['dataset'], name)
        print(json.dumps(dict(stage=stage, output=str(output), config='PASS', sensor_forward='SERVER_VALIDATION_REQUIRED')))
        return
    if int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('this trainer supports one process per experiment')
    if output.exists() and not opt.resume:
        raise FileExistsError('new experiment output already exists: '+str(output))
    from opencood.hypes_yaml.yaml_utils import load_general_params
    from opencood.models.codefilling_doma import CodefillingDoma
    from opencood.data_utils.codefilling_dataset import build_codefilling_dataset
    cfg = load_general_params(cfg)
    seed_all(cfg['seed'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    model = CodefillingDoma(cfg['model']['args']).to(opt.device)
    provenance = initialize_parents(model, cfg, parents) if not opt.resume else {}
    args = cfg['optimizer']
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(args['lr']),
                                 eps=float(args['args'].get('eps', 1e-8)),
                                 weight_decay=float(args['args'].get('weight_decay', 0)))
    sched = cfg['lr_scheduler']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched['step_size'], gamma=sched['gamma'])
    start, best = 0, float('inf')
    if opt.resume:
        state = resume_checkpoint(opt.resume, model, optimizer, scheduler, cfg, generator)
        start, best, provenance = state['epoch']+1, state['best'], state['parents']
    criterion = None
    if stage != 'codebook':
        from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
        criterion = PointPillarDepthLoss(cfg['loss']['args']).to(opt.device)
    train_set = build_codefilling_dataset(cfg, train=True)
    val_set = build_codefilling_dataset(cfg, train=False)
    # num_workers=0 is portable on Windows and makes epoch-boundary RNG restoration exact.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['train_params']['batch_size'],
        shuffle=True, num_workers=0, generator=generator, collate_fn=train_set.collate_batch_train)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg['train_params']['batch_size'],
        shuffle=False, num_workers=0, generator=generator, collate_fn=val_set.collate_batch_train)
    # Dataset constructors/reinitialize consume RNG. Restore only after they finish.
    if opt.resume:
        from opencood.utils.codefilling_state import restore_rng
        restore_rng(state['rng'], generator)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    for epoch in range(start, cfg['train_params']['epoches']):
        # Rebuild shuffled CAV order deterministically for this epoch, independently
        # from augmentation/optimizer RNG and from constructor side effects.
        import random
        import numpy as np
        py_state, np_state = random.getstate(), np.random.get_state()
        random.seed(cfg['seed'] + epoch)
        np.random.seed(cfg['seed'] + epoch)
        train_set.reinitialize()
        random.setstate(py_state)
        np.random.set_state(np_state)
        model.train()
        train_losses = []
        for batch in train_loader:
            if batch is None:
                raise RuntimeError('dataset returned None; audit data rather than silently dropping a frame')
            optimizer.zero_grad(set_to_none=True)
            loss = stage_loss(model, to_device(batch, opt.device), criterion, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        values = []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    raise RuntimeError('validation sample missing')
                values.append(float(stage_loss(model, to_device(batch, opt.device), criterion, cfg)))
        if not values or not train_losses or not all(__import__('math').isfinite(v) for v in values):
            raise RuntimeError('empty/nonfinite epoch')
        value = sum(values)/len(values)
        improved = value < best
        best = min(best, value)
        scheduler.step()
        save_checkpoint(output/'last.pth', model, optimizer, scheduler, cfg, epoch, best, generator, provenance)
        if improved:
            save_checkpoint(output/'best.pth', model, optimizer, scheduler, cfg, epoch, best, generator, provenance, True)
        with (output/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(epoch=epoch, train_loss=sum(train_losses)/len(train_losses), val_loss=value, best=best))+'\n')


if __name__ == '__main__':
    main()
