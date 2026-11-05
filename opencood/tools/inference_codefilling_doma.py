"""Fixed m1 ego, same-frame CAV1..4 evaluation with native AP and a byte ledger."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import torch
import yaml
from opencood.utils.codefilling_protocol import (load_config, digest,
                                                experiment_output, validate_evaluation_frame, evaluation_sample_key)
from opencood.utils.codefilling_state import seed_all, read_checkpoint
from opencood.tools.train_codefilling_doma import to_device


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seeded_evaluation_batches(loader, seed):
    # Imports, model/dataset construction, and DataLoader's iterator creation may
    # consume RNG. Start the native sequential Gumbel stream only after all three.
    iterator = iter(loader)
    seed_all(seed)
    yield from iterator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', help='optional explicit canonical stage path under opencood/logs/CodeFilling_*_SEED303/')
    parser.add_argument('--device', default='cuda')
    opt = parser.parse_args()
    cfg = load_config(opt.config)
    if cfg['codefilling_protocol']['stage'] != 'infer':
        raise ValueError('requires infer.yaml')
    checkpoint = read_checkpoint(opt.checkpoint, cfg['fusion']['dataset'], 'finetune')
    from opencood.hypes_yaml.yaml_utils import load_general_params
    from opencood.models.codefilling_doma import CodefillingDoma
    from opencood.data_utils.codefilling_dataset import build_codefilling_dataset
    from opencood.utils import eval_utils
    cfg = load_general_params(cfg)
    output = experiment_output(cfg, opt.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    with (output/'run.json').open('w', encoding='utf-8') as stream:
        json.dump(dict(config_path=str(Path(opt.config).resolve()), checkpoint_sha256=digest(opt.checkpoint),
                       sampling='native hard Gumbel; seed303 after setup per CAV group; no per-frame reseed'), stream, indent=2)
    seed_all(cfg['seed'])
    model = CodefillingDoma(cfg['model']['args']).to(opt.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    results, baseline = [], None
    for count in (1, 2, 3, 4):
        seed_all(cfg['seed'])
        current = copy.deepcopy(cfg)
        current['use_cav'] = count
        dataset = build_codefilling_dataset(current, train=False, inference=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
            shuffle=False, collate_fn=dataset.collate_batch_test,
            generator=torch.Generator().manual_seed(cfg['seed']))
        stats = {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}
        frame_ledger = []
        with (output/('cav'+str(count)+'_frames.jsonl')).open('w', encoding='utf-8') as log:
            for index, batch in enumerate(seeded_evaluation_batches(loader, cfg['seed'])):
                if batch is None:
                    raise RuntimeError('missing test frame '+str(index))
                batch = to_device(batch, opt.device)
                with torch.no_grad():
                    prediction = model(batch['ego'])
                    boxes, scores, gt = dataset.post_process(batch, {'ego': prediction})
                for threshold in stats:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, threshold)
                mods = batch['ego']['agent_modality_list']
                ledger = prediction['communication']
                row = dict(index=index, sample_idx=batch['ego']['sample_idx'],
                           sample_key=evaluation_sample_key(dataset, index),
                           all_cav_ids=batch['ego']['cav_id_list'], input_cav_ids=batch['ego']['cav_id_list'][:len(mods)],
                           modalities=mods, requested_cavs=count, realized_cavs=len(mods), gt_hash=tensor_hash(gt), communication=ledger)
                identity = validate_evaluation_frame(row)
                log.write(json.dumps(row, default=lambda x: x.item() if hasattr(x,'item') else str(x))+'\n')
                frame_ledger.append((identity, ledger['total_bytes'], len(mods)))
        identity = [entry[0] for entry in frame_ledger]
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise ValueError('CAV1..4 evaluated different frames or GT')
        if not identity or any(s['gt'] == 0 for s in stats.values()):
            raise ValueError('empty evaluation/GT; cannot report AP')
        aps = {f'AP@{t:.1f}': float(eval_utils.calculate_ap(stats, t)[0]) for t in stats}
        summary = dict(dataset=cfg['fusion']['dataset'], requested_cavs=count, frames=len(identity),
                       mean_bytes=sum(row[1] for row in frame_ledger)/len(identity),
                       short_prefix_frames=sum(row[2] < count for row in frame_ledger), **aps)
        results.append(summary)
        with (output/('cav'+str(count)+'_raw_ap.json')).open('w',encoding='utf-8') as stream:
            json.dump(stats, stream, indent=2, default=lambda x: float(x))
    with (output/'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (output/'results.json').open('w', encoding='utf-8') as stream:
        json.dump(dict(results=results, checkpoint_sha256=digest(opt.checkpoint), effective_config=cfg), stream, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
