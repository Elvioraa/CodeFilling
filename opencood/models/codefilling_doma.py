"""Four-member assembly of native CodeFilling, independent from DOMA networks.

Native shared-head/solver feature flow: encoder -> member backbone -> aligner ->
UMGM (ego bypass) -> original Information Filling mask -> multiscale max fusion
-> shrink/head. Only physical crop/pad and protocol checks are added.
"""
from collections import Counter
import copy
import torch
from torch import nn
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.fuse_modules.codefilling_max import MaxFusion
from opencood.models.sub_modules.codefilling_transport import CodeFillingTransport, center_grid


class CodeFillingCore(nn.Module):
    """Sensor-independent real CodeFilling path; exercised on Windows CPU."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stage = args['stage']
        self.transport = CodeFillingTransport(args['codebook'], args['fusion_args']['communication'])
        self.backbone = ResNetBEVBackbone(args['fusion_backbone'])
        self.fusion_net = nn.ModuleList(MaxFusion() for _ in args['fusion_backbone']['layer_nums'])
        self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.cls_head_single = nn.Conv2d(64, args['anchor_number'], 1)
        self.reg_head_single = nn.Conv2d(64, 7*args['anchor_number'], 1)
        self.dir_head_single = nn.Conv2d(64, args['dir_args']['num_bins']*args['anchor_number'], 1)
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], 1)
        self.reg_head = nn.Conv2d(args['in_head'], 7*args['anchor_number'], 1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins']*args['anchor_number'], 1)

    def forward_features(self, x, record_len, transforms):
        single = dict(cls_preds_single=self.cls_head_single(x), reg_preds_single=self.reg_head_single(x),
                      dir_preds_single=self.dir_head_single(x))
        if self.stage == 'codebook':
            _, loss = self.transport.reconstruct(x.detach(), record_len)
            return dict(codebook_loss=loss)
        confidence = torch.split(single['cls_preds_single'], list(map(int, record_len)))
        enabled = self.stage in ('finetune', 'infer')
        if enabled:
            x, masks, rates, cb_loss, ledger = self.transport(x, confidence, record_len, transforms,
                coarse_channels=self.args['fusion_backbone']['num_filters'][1:])
        else:
            # Detector learning precedes communication restriction and quantization.
            masks = x.new_ones((len(x), 1, *x.shape[-2:]))
            rates, cb_loss, ledger = x.new_zeros(()), x.sum()*0, None
        features = [x]
        for i in range(1, len(self.fusion_net)):
            x = self.backbone.get_layer_i_feature(x, layer_i=i)
            features.append(x)
        fused = []
        for i, fuse in enumerate(self.fusion_net):
            fused.append(fuse(features[i]*masks, record_len, transforms))
            if i+1 < len(features):
                masks = torch.nn.functional.max_pool2d(masks, 2)
        x = self.shrink_conv(self.backbone.decode_multiscale_feature(fused))
        single.update(cls_preds=self.cls_head(x), reg_preds=self.reg_head(x), dir_preds=self.dir_head(x),
                      comm_rates=rates, codebook_loss=cb_loss, communication=ledger)
        return single

    def configure_stage(self):
        for name, parameter in self.named_parameters():
            is_cb = name.startswith('transport.quantizer.')
            parameter.requires_grad_(is_cb if self.stage == 'codebook' else not is_cb and self.stage != 'infer')
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == 'codebook':
            # No BN running-stat drift in frozen perception modules.
            for name, module in self.named_children():
                module.eval()
            self.transport.quantizer.train(mode)
        elif self.stage in ('finetune', 'infer'):
            self.transport.quantizer.eval()
        return self


class CodefillingDoma(CodeFillingCore):
    def __init__(self, args):
        super().__init__(args)
        # Sensor dependencies are needed only for real datasets, not CPU contracts.
        from opencood.models import heter_encoders
        from opencood.models.sub_modules.feature_alignnet import AlignNet
        encoders = {'point_pillar': heter_encoders.PointPillar, 'second': heter_encoders.SECOND,
                    'lift_splat_shoot': heter_encoders.LiftSplatShoot}
        self.modalities = [m for m in ('m1', 'm2', 'm3', 'm4') if m in args]
        for m in self.modalities:
            setting = args[m]
            # Native PointPillar inserts a NumPy grid_size into its input dict.
            # Keep the signed experiment config immutable/serializable.
            encoder_args = copy.deepcopy(setting['encoder_args'])
            if setting['sensor_type'] == 'camera':
                encoder_args['camera_contract'] = 'doma'
            setattr(self, 'encoder_'+m, encoders[setting['core_method']](encoder_args))
            setattr(self, 'backbone_'+m, ResNetBEVBackbone(setting['backbone_args']))
            setattr(self, 'aligner_'+m, AlignNet(setting['aligner_args']))
        self.configure_stage()

    def forward(self, data):
        from opencood.utils.transformation_utils import normalize_pairwise_tfm
        order = data['agent_modality_list']
        counts = Counter(order)
        bounds = self.args['lidar_range']
        height, width = round((bounds[4]-bounds[1])/.8), round((bounds[3]-bounds[0])/.8)
        feature_dict, depths = {}, {}
        for m in self.modalities:
            if not counts[m]:
                continue
            encoder = getattr(self, 'encoder_'+m)
            x = encoder(data, m)
            x = getattr(self, 'backbone_'+m)({'spatial_features': x})['spatial_features_2d']
            x = getattr(self, 'aligner_'+m)(x)
            feature_dict[m] = center_grid(x, height, width)
            if self.args[m]['encoder_args'].get('depth_supervision'):
                depths['depth_items_'+m] = encoder.depth_items
        index = Counter()
        packed = []
        for m in order:
            packed.append(feature_dict[m][index[m]])
            index[m] += 1
        transforms = normalize_pairwise_tfm(data['pairwise_t_matrix'], bounds[4]-bounds[1], bounds[3]-bounds[0], 1)
        result = self.forward_features(torch.stack(packed), data['record_len'], transforms)
        result.update(depths)
        return result
