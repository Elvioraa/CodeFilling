"""Protocol-only dataset factory; the official dataset registry remains unchanged."""
import copy
from pathlib import Path
from opencood.utils.codefilling_protocol import MEMBERS, ordered_cavs, lidar_path


def build_codefilling_dataset(cfg, train=False, inference=False):
    from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
    from opencood.data_utils.datasets.basedataset.v2xset_basedataset import V2XSETBaseDataset
    from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset
    from opencood.data_utils.datasets.codefilling_infer_dataset import getIntermediateheterinferFusionDataset
    from opencood.utils.heter_utils import Adaptor

    cfg = copy.deepcopy(cfg)
    ds = cfg['fusion']['dataset']
    base = {'opv2v': OPV2VBaseDataset, 'v2xset': V2XSETBaseDataset}[ds]
    factory = getIntermediateheterinferFusionDataset if inference else getIntermediateheterFusionDataset
    parent = factory(base)

    class ProtocolAdaptor(Adaptor):
        def reorder_cav_list(self, cavs, scenario):
            if inference:
                return ordered_cavs(cavs, self.modality_assignment[scenario])
            return super().reorder_cav_list(cavs, scenario)

        def switch_lidar_channels(self, modality, path):
            # Strict no-fallback policy for OPV2V m3 and raw V2XSet m1/m3.
            path = lidar_path(path, ds, modality)
            if not Path(path).is_file():
                raise FileNotFoundError('SERVER_VALIDATION_REQUIRED: missing PCD '+path)
            return path

    class ProtocolDataset(parent):
        def reinitialize(self):
            # Same evaluation sample universe across all stages and CAV counts.
            if ds == 'v2xset':
                self.scenario_folders = [p for p in self.scenario_folders if '2021_09_09_13_20_58' not in p]
            if hasattr(self, 'adaptor'):
                self.adaptor.__class__ = ProtocolAdaptor
            super().reinitialize()

        def __getitem__(self, index):
            item = super().__getitem__(index)
            if inference and item is not None:
                ego = item['ego']
                mods = ego['agent_modality_list']
                # Out-of-range slots must never silently relabel another member.
                if mods != list(MEMBERS[:len(mods)]):
                    raise ValueError('non-prefix members after communication-range filtering: '+str(mods))
            return item

    # DOMA ordered evaluation uses visualize=True to force LiDAR GT for all slots.
    return ProtocolDataset(cfg, visualize=inference, train=train)
