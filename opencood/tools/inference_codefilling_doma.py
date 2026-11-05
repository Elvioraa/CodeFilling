"""Fixed m1 ego, same-frame CAV1..4 evaluation with native AP and a byte ledger."""
import argparse
import contextlib
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import uuid
import torch
import yaml
from opencood.utils.codefilling_protocol import (load_config, digest,
    experiment_output, validate_evaluation_frame, evaluation_sample_key, fingerprint)
from opencood.utils.codefilling_state import seed_all, read_checkpoint
from opencood.tools.train_codefilling_doma import to_device


DOMA_EVALUATION_BOUNDS = (-204.8, -102.4, -3, 204.8, 102.4, 1)
DOMA_COMMUNICATION_RANGE = 180


def apply_doma_evaluation_protocol(cfg, enabled=False):
    """Opt-in runtime geometry; preserve camera grids and the disabled path."""
    if not enabled:
        return cfg
    cfg = copy.deepcopy(cfg)
    bounds = list(DOMA_EVALUATION_BOUNDS)
    cfg['cav_lidar_range'] = bounds
    cfg['preprocess']['cav_lidar_range'] = bounds
    cfg['postprocess']['gt_range'] = bounds
    cfg['postprocess']['anchor_args']['cav_lidar_range'] = bounds
    cfg['model']['args']['lidar_range'] = bounds
    for modality in ('m1', 'm3'):
        cfg['heter']['modality_setting'][modality]['preprocess']['cav_lidar_range'] = bounds
        cfg['model']['args'][modality]['encoder_args']['lidar_range'] = bounds
    for modality in ('m2', 'm4'):
        cfg['model']['args'][modality]['camera_mask_args']['cav_lidar_range'] = bounds
    # Communication eligibility is independent of the detection grid extents.
    cfg['comm_range'] = DOMA_COMMUNICATION_RANGE
    cfg['codefilling_protocol']['evaluation_protocol'] = dict(
        name='doma', eval_bounds=bounds, comm_range=DOMA_COMMUNICATION_RANGE)
    return cfg


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seeded_evaluation_batches(loader, seed):
    # Imports, model/dataset construction, and DataLoader's iterator creation may
    # consume RNG. Start the native sequential Gumbel stream only after all three.
    iterator = iter(loader)
    seed_all(seed)
    yield from iterator


EVAL_FIELDS = ('ap30', 'ap_50', 'ap_70', 'mpre_50', 'mrec_50', 'mpre_70', 'mrec_70')


def evaluation_values(stats, calculate_ap):
    """The same seven fields and curves as DOMA eval_final_results."""
    ap30, _, _ = calculate_ap(stats, .3)
    ap50, recall50, precision50 = calculate_ap(stats, .5)
    ap70, recall70, precision70 = calculate_ap(stats, .7)
    return dict(ap30=float(ap30), ap_50=float(ap50), ap_70=float(ap70),
                mpre_50=[float(v) for v in precision50], mrec_50=[float(v) for v in recall50],
                mpre_70=[float(v) for v in precision70], mrec_70=[float(v) for v in recall70])


def complete_evaluation(values):
    if not isinstance(values, dict) or not all(k in values for k in EVAL_FIELDS):
        return False
    try:
        if any(not math.isfinite(float(values[k])) or not 0 <= float(values[k]) <= 1
               for k in ('ap30', 'ap_50', 'ap_70')):
            return False
        for threshold in ('50', '70'):
            precision, recall = values['mpre_'+threshold], values['mrec_'+threshold]
            if not isinstance(precision, list) or not isinstance(recall, list) or len(precision) != len(recall) or len(recall) < 2:
                return False
            if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in precision+recall):
                return False
        return True
    except (TypeError, ValueError):
        return False


def frame_modalities(cfg, assignment, key, ids):
    return [cfg['heter']['mapping_dict'][assignment[key['scene']][str(cav)]] for cav in ids]


def signature(identities):
    return dict(frames=len(identities), frame_digest=fingerprint(identities))


def reusable_cav1(output, cfg, context, assignment, calculate_ap):
    """Read-only cache inspection. Never treat a partial frame log as completed AP."""
    output = Path(output)
    if not output.is_dir():
        return None
    try:
        # New-format runs keep context and frame/GT signature inside infer.log.
        log = output/'infer.log'
        logged_context, completed = None, []
        if log.is_file():
            for line in log.read_text(encoding='utf-8').splitlines():
                if line.startswith('CODEFILLING_RUN '):
                    logged_context = json.loads(line.split(' ', 1)[1])
                    completed = []
                elif line.startswith('CAV_COMPLETE ') and logged_context == context:
                    try:
                        completed.append(json.loads(line.split(' ', 1)[1]))
                    except ValueError:
                        # A killed later group must not invalidate completed CAV1.
                        continue
            for entry in reversed(completed):
                if entry.get('requested_cavs') != 1 or entry.get('frames', 0) <= 0 or not entry.get('frame_digest'):
                    continue
                path = output/entry['eval_file']
                if path.parent.resolve() != output.resolve() or digest(path) != entry['eval_sha256']:
                    continue
                values = load_config(path)
                if complete_evaluation(values):
                    return dict(values={k: values[k] for k in EVAL_FIELDS},
                                signature={k: entry[k] for k in ('frames','frame_digest')}, source=path.name)
        # Legacy runs can have completed CAV1 raw AP before a CAV2 failure.
        run = json.loads((output/'run.json').read_text(encoding='utf-8'))
        if (run.get('checkpoint_sha256') != context['checkpoint_sha256'] or
                fingerprint(load_config(output/'config.yaml')) != context['config_sha256']):
            return None
        rows = [json.loads(line) for line in (output/'cav1_frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        identities = []
        for index, row in enumerate(rows):
            if row.get('index') != index or row.get('sample_idx') != index or row.get('requested_cavs') != 1:
                return None
            row['cav_modalities'] = frame_modalities(cfg, assignment, row['sample_key'], row['all_cav_ids'])
            identities.append(validate_evaluation_frame(row))
        if not identities:
            return None
        for path in sorted(output.glob('eval_intermediate_*use_cav1.yaml')):
            values = load_config(path)
            if complete_evaluation(values):
                return dict(values={k: values[k] for k in EVAL_FIELDS},
                            signature=signature(identities), source=path.name)
        raw = json.loads((output/'cav1_raw_ap.json').read_text(encoding='utf-8'))
        stats = {float(k): v for k, v in raw.items()}
        for threshold in (.3, .5, .7):
            stat = stats[threshold]
            if stat['gt'] <= 0 or len(stat['tp']) != len(stat['fp']) or len(stat['tp']) != len(stat['score']):
                return None
        values = evaluation_values(stats, calculate_ap)
        if complete_evaluation(values):
            return dict(values=values, signature=signature(identities), source='cav1_raw_ap.json')
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        # Leave unverifiable/incomplete history intact for the backup below.
        return None
    return None


def prepare_output(output):
    """Move nonempty history to a unique sibling backup; never remove its files."""
    output = Path(output)
    if output.is_symlink():
        raise ValueError('infer output must not be a symlink')
    backup = None
    if output.exists() and any(output.iterdir()):
        backup = output.with_name(output.name+'_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        # Verify both resolved targets before moving a directory on Windows.
        if backup.resolve().parent != output.resolve().parent or backup.exists():
            raise ValueError('invalid infer backup destination')
        output.rename(backup)
    output.mkdir(parents=True, exist_ok=True)
    return backup


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def evaluate_group(cfg, count, model, device, assignment, build_dataset, eval_utils, reuse=None):
    seed_all(cfg['seed'])
    current = copy.deepcopy(cfg)
    current['use_cav'] = count
    dataset = build_dataset(current, train=False, inference=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
        shuffle=False, collate_fn=dataset.collate_batch_test,
        generator=torch.Generator().manual_seed(cfg['seed']))
    stats = {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}
    identities, total_bytes, short_frames, gt_count = [], 0, 0, 0
    for index, batch in enumerate(seeded_evaluation_batches(loader, cfg['seed'])):
        if batch is None:
            raise RuntimeError('missing test frame '+str(index))
        batch = to_device(batch, device)
        with torch.no_grad():
            if reuse is not None:
                # Same GT function as Dataset.post_process, without CAV1 forward.
                gt = dataset.post_processor.generate_gt_bbx(batch)
                ledger = dict(peers=0, messages=[], total_bytes=0)
            else:
                prediction = model(batch['ego'])
                boxes, scores, gt = dataset.post_process(batch, {'ego': prediction})
                ledger = prediction['communication']
                for threshold in stats:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, threshold)
        mods, ids = batch['ego']['agent_modality_list'], batch['ego']['cav_id_list']
        key = evaluation_sample_key(dataset, index)
        row = dict(sample_idx=batch['ego']['sample_idx'], sample_key=key, all_cav_ids=ids,
                   input_cav_ids=ids[:len(mods)], modalities=mods,
                   cav_modalities=frame_modalities(cfg, assignment, key, ids), requested_cavs=count,
                   realized_cavs=len(mods), gt_hash=tensor_hash(gt), communication=ledger)
        identities.append(validate_evaluation_frame(row))
        total_bytes += ledger['total_bytes']
        short_frames += len(mods) < count
        gt_count += len(gt)
        print(f'CAV{count} {"GT-check/reuse" if reuse else "eval"} {index+1}/{len(dataset)} '
              f'CAV IDs={row["input_cav_ids"]} modalities={mods} bytes={ledger["total_bytes"]}', flush=True)
    if len(identities) != len(dataset) or not identities or gt_count == 0:
        raise ValueError('empty/incomplete evaluation or GT')
    checked = signature(identities)
    if reuse is not None and checked != reuse['signature']:
        print('CAV1 cached frame/GT signature differs; preserving backup and evaluating current CAV1.', flush=True)
        return evaluate_group(cfg, count, model, device, assignment, build_dataset, eval_utils)
    values = reuse['values'] if reuse is not None else evaluation_values(stats, eval_utils.calculate_ap)
    if not complete_evaluation(values):
        raise ValueError('nonfinite/incomplete AP evaluation')
    return values, dict(requested_cavs=count, **checked, mean_bytes=total_bytes/len(identities),
                        short_prefix_frames=short_frames, reused=reuse is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', help='optional explicit canonical stage path under opencood/logs/CodeFilling_*_SEED303/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--doma-evaluation-protocol', action='store_true',
                        help='use DOMA evaluation bounds and independent 180m communication range')
    opt = parser.parse_args()
    cfg = load_config(opt.config)
    if cfg['codefilling_protocol']['stage'] != 'infer':
        raise ValueError('requires infer.yaml')
    cfg = apply_doma_evaluation_protocol(cfg, opt.doma_evaluation_protocol)
    cfg['validate_dir'] = cfg['test_dir']
    checkpoint = read_checkpoint(opt.checkpoint, cfg['fusion']['dataset'], 'finetune')
    from opencood.hypes_yaml.yaml_utils import load_general_params
    from opencood.models.codefilling_doma import CodefillingDoma
    from opencood.data_utils.codefilling_dataset import build_codefilling_dataset
    from opencood.utils import eval_utils
    cfg = load_general_params(cfg)
    output = experiment_output(cfg, opt.output)
    assignment = json.loads(Path(cfg['heter']['assignment_path']).read_text(encoding='utf-8'))
    context = dict(checkpoint_sha256=digest(opt.checkpoint), config_sha256=fingerprint(cfg),
                   assignment_sha256=digest(cfg['heter']['assignment_path']))
    reuse = reusable_cav1(output, cfg, context, assignment, eval_utils.calculate_ap)
    backup = prepare_output(output)
    with (output/'infer.log').open('w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            print('CODEFILLING_RUN '+json.dumps(context), flush=True)
            print(f'config={Path(opt.config).resolve()} checkpoint={Path(opt.checkpoint).resolve()} backup={backup}', flush=True)
            print('sampling=native hard Gumbel; seed303 after setup per CAV group; no per-frame reseed', flush=True)
            print(f'CAV1 reusable source={reuse["source"] if reuse else "none (missing/incomplete/incompatible history)"}', flush=True)
            seed_all(cfg['seed'])
            model = CodefillingDoma(cfg['model']['args']).to(opt.device).eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            baseline = None
            for count in (1,2,3,4):
                values, summary = evaluate_group(cfg, count, model, opt.device, assignment,
                    build_codefilling_dataset, eval_utils, reuse if count == 1 else None)
                checked = {k: summary[k] for k in ('frames','frame_digest')}
                if baseline is None:
                    baseline = checked
                elif checked != baseline:
                    raise ValueError('CAV1..4 evaluated different frames or GT')
                path = output/f'eval_intermediate_use_cav{count}.yaml'
                with path.open('w', encoding='utf-8') as stream:
                    yaml.safe_dump(values, stream, sort_keys=False)
                summary.update(eval_file=path.name, eval_sha256=digest(path),
                               AP30=values['ap30'], AP50=values['ap_50'], AP70=values['ap_70'])
                print('CAV_COMPLETE '+json.dumps(summary), flush=True)
        except BaseException:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
