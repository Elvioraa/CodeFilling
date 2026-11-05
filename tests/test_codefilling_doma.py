"""CPU contracts use real UMGM/Information Filling/BEV fusion, never fake AP."""
import ast
import copy
import json
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from opencood.utils.codefilling_protocol import ROOT, MEMBERS, load_config, validate_config, lidar_path, ordered_cavs, apply_server_settings, fingerprint
from opencood.utils.codefilling_protocol import experiment_output, validate_evaluation_frame, evaluation_sample_key, validate_input_members
from opencood.models.sub_modules.codefilling_transport import (CodeFillingTransport, make_quantizer, communication_ledger,
    center_grid, pack_indices, unpack_indices, decode_sampled_codes)
from opencood.models.comm_modules.where2comm import Communication
from opencood.models.codefilling_doma import CodeFillingCore
from opencood.tools.check_codefilling_doma import check_static, audit_data
from opencood.tools.prepare_codefilling_doma import put
from opencood.utils.codefilling_state import (seed_all, initialize_parents, save_checkpoint, resume_checkpoint, SCHEMA)

torch.set_num_threads(1)


def config(ds='opv2v', stage='infer'):
    return load_config(ROOT/('opencood/hypes_yaml/'+ds+'/doma_codefilling/'+stage+'.yaml'))


def identity(count, batch=1):
    return torch.eye(2, 3)[None,None,None].repeat(batch,count,count,1,1)


@pytest.mark.parametrize('ds', ['opv2v','v2xset'])
def test_config_sources_stages_and_negative_contracts(ds):
    assert check_static(ds)['status'] == 'PASS_STATIC'
    cfg = config(ds)
    cfg['model']['args']['m4']['encoder_args']['camera_encoder'] = 'EfficientNet'
    with pytest.raises(ValueError, match='camera trunk'):
        validate_config(cfg)
    cfg = config(ds)
    cfg['heter']['lidar_channels_dict'] = {'m1':32}
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = config(ds)
    cfg['heter']['mapping_dict']['m3'] = 'm1'
    with pytest.raises(ValueError, match='four modalities'):
        validate_config(cfg)


def test_paths_order_and_non_overwrite(tmp_path):
    assert lidar_path(r'D:\dataset\OPV2V\test\s\1\0001.pcd','opv2v','m3') == 'D:/dataset/OPV2V_Hetero/test/s/1/0001_32.pcd'
    for m in ('m1','m3'):
        assert lidar_path('/dataset/V2XSET/test/1.pcd','v2xset',m).endswith('/1.pcd')
    assert ordered_cavs(['3','-1','1','2'], {'1':'m1','2':'m2','3':'m3','-1':'m4'}) == ['1','2','3','-1']
    with pytest.raises(ValueError):
        ordered_cavs(['1','2'], {'1':'m2','2':'m1'})
    path = tmp_path/'keep.yaml'
    put(path,'original\n')
    put(path,'original\n')
    with pytest.raises(FileExistsError):
        put(path,'replacement\n')
    assert path.read_text() == 'original\n'


def test_server_receipt_is_explicit(tmp_path):
    with pytest.raises(ValueError, match='SERVER_VALIDATION_REQUIRED'):
        apply_server_settings(config(), ROOT/'docs/server_settings.example.yaml')
    path = tmp_path/'receipt.json'
    settings = dict(status='VERIFIED', evidence='unit fixture only', dataset='v2xset',
                    eval_range=[-12.8,-6.4,-3,12.8,6.4,1],comm_range=70,test_dir='fixture/test')
    settings['doma_formal_evaluation'] = dict(status='VERIFIED',command='unit fixture, not a real server command',
        evidence='unit fixture only',eval_range=settings['eval_range'],comm_range=70,test_dir='fixture/test',
        excluded_scenes=['2021_09_09_13_20_58'],scene_policy_evidence='unit fixture only')
    path.write_text(json.dumps(settings))
    cfg = apply_server_settings(config('v2xset'),path)
    assert cfg['model']['args']['m3']['encoder_args']['lidar_range'] == [-12.8,-6.4,-3,12.8,6.4,1]
    assert cfg['heter']['lidar_channels_dict'] == {}
    for field in ('command','scene_policy_evidence'):
        invalid = copy.deepcopy(settings)
        invalid['doma_formal_evaluation'][field] = None
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError,match='SERVER_VALIDATION_REQUIRED'):
            apply_server_settings(config('v2xset'),path)
    for field, value in [('eval_range',[-25.6,-6.4,-3,25.6,6.4,1]),('comm_range',180),
                         ('test_dir','other/test'),('excluded_scenes',[])]:
        invalid = copy.deepcopy(settings)
        invalid['doma_formal_evaluation'][field] = value
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError,match='SERVER_VALIDATION_REQUIRED'):
            apply_server_settings(config('v2xset'),path)


def test_native_quantizer_loss_samples_and_wire_roundtrip():
    cb = dict(seg_num=2,dict_size=7,r=3)
    transport = CodeFillingTransport(cb,dict(thre=.2,solver=dict(thre=1.,method='sum')))
    native = make_quantizer(cb)
    native.load_state_dict(transport.quantizer.state_dict(),strict=True)
    x = torch.randn(5,64,4,4,requires_grad=True)
    seed_all(91)
    expected,_,_,loss = native(x.permute(0,2,3,1).reshape(-1,64))
    seed_all(91)
    actual,actual_loss = transport.reconstruct(x,torch.tensor([2,3]))
    expected = expected.reshape(5,4,4,64).permute(0,3,1,2).contiguous()
    expected = torch.where(torch.tensor([True,False,True,False,False])[:,None,None,None], x, expected)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual_loss,loss,rtol=0,atol=0)
    restored = decode_sampled_codes(transport.quantizer,transport.last_codes).reshape(5,4,4,64).permute(0,3,1,2)
    torch.testing.assert_close(restored[[1,3,4]],actual[[1,3,4]],rtol=1e-6,atol=1e-6)
    for code in transport.last_codes:
        flat = code.flatten().tolist()
        packed = pack_indices(flat,7)
        assert len(packed) == math.ceil(len(flat)*3/8)
        assert unpack_indices(packed,len(flat),7) == flat
    actual_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in transport.quantizer.parameters())
    with pytest.raises(ValueError):
        pack_indices([7],7)


@pytest.mark.parametrize('method',['sum','max'])
def test_original_information_filling_mask_and_single_ego(method):
    args = dict(thre=.2,solver=dict(thre=1.,method=method))
    communication = Communication(args)
    confidence = torch.logit(torch.tensor([.1,.8,.6,.3]))[:,None,None,None].expand(4,2,4,4)
    _,mask,rate = communication([confidence],torch.tensor([4]),identity(4))
    assert torch.all(mask[0] == 1)
    assert torch.all(mask[1] == 1)
    assert torch.all(mask[2:] == 0)
    assert rate.item() == pytest.approx(1/3)
    _,single,single_rate = communication([confidence[:1]],torch.tensor([1]),identity(1))
    ledger = communication_ledger(single,[1],dict(seg_num=2,dict_size=64,r=3),coarse_channels=[128,256])
    assert single_rate.item() == 0 and ledger['total_bytes'] == 0


def test_multiscale_bytes_fractional_masks_and_batch_exclusion():
    masks = torch.zeros(5,1,4,4)
    masks[[0,2]] = 1  # local egos never count
    masks[1,0,0,0] = .5
    masks[3,0,1,1] = 1
    cfg = dict(seg_num=2,dict_size=64,r=3)
    ledger = communication_ledger(masks,[2,3],cfg,coarse_channels=[128,256])
    assert ledger['peers'] == 3 and ledger['selected_cells'] == 2
    assert ledger['messages'][0]['index_bits'] == 36
    assert ledger['messages'][0]['index_bytes'] == len(pack_indices([0]*6,64)) == 5
    assert ledger['messages'][0]['mask_weight_bytes'] == 4
    assert ledger['messages'][0]['coarse_bytes'] == (128+256)*4
    assert ledger['feature_bytes'] == (5+2+4+1536)+(5+2+1536)+2
    assert ledger['control_bytes'] == 3*(4*4*4+6*4)
    assert ledger['total_bytes'] == ledger['control_bytes']+ledger['feature_bytes']
    assert ledger['measured_network_bytes'] is None


def test_physical_crop_pad_preserves_grid_without_resizing():
    x = torch.arange(16.).reshape(1,1,4,4)
    padded = center_grid(x,8,8)
    torch.testing.assert_close(padded[:,:,2:6,2:6],x)
    assert padded.count_nonzero() == x.count_nonzero()
    torch.testing.assert_close(center_grid(padded,4,4),x)


def core_args(stage):
    args = config()['model']['args']
    args['stage'] = stage
    args['fusion_backbone']['layer_nums'] = [1,1,1]
    args['codebook'] = dict(seg_num=2,dict_size=8,r=2)
    return args


@pytest.mark.parametrize('stage',['detector','codebook','finetune','infer'])
def test_real_cpu_core_forward_backward_and_freeze(stage):
    model = CodeFillingCore(core_args(stage))
    model.configure_stage()
    model.train(stage != 'infer')
    before = {n:t.clone() for n,t in model.state_dict().items()}
    output = model.forward_features(torch.randn(4,64,8,8),torch.tensor([4]),identity(4))
    if stage == 'codebook':
        loss = output['codebook_loss']
    else:
        assert output['cls_preds'].shape == (1,2,8,8)
        loss = output['cls_preds'].square().mean()+output['reg_preds'].square().mean()
    if stage != 'infer':
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        loss.backward()
        optimizer.step()
        if stage == 'codebook':
            assert all(torch.equal(before[n],t) for n,t in model.state_dict().items() if not n.startswith('transport.quantizer.'))
        if stage == 'finetune':
            assert all(torch.equal(before[n],t) for n,t in model.state_dict().items() if n.startswith('transport.quantizer.'))
    else:
        assert not any(p.requires_grad for p in model.parameters())
        assert output['communication']['peers'] == 3


def test_epoch_resume_restores_next_update_and_rng(tmp_path):
    seed_all(303)
    cfg = config(stage='pretrain_m1')
    model = nn.Sequential(nn.Linear(3,4),nn.Dropout(.2),nn.Linear(4,1))
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[1,3],gamma=.5)
    generator = torch.Generator().manual_seed(31)
    def step():
        optimizer.zero_grad()
        loss = model(torch.randn(5,3)).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach().clone(),torch.rand(3,generator=generator)
    step()
    path = tmp_path/'last.pth'
    save_checkpoint(path,model,optimizer,scheduler,cfg,0,1.,generator,{})
    expected = step()
    state = copy.deepcopy(model.state_dict())
    resume_checkpoint(path,model,optimizer,scheduler,cfg,generator)
    actual = step()
    for x,y in zip(actual,expected):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in state.items())
    cfg['seed'] = 999
    with pytest.raises(ValueError,match='mismatch'):
        resume_checkpoint(path,model,optimizer,scheduler,cfg,generator)


def test_parent_ownership_and_fail_closed(tmp_path):
    class Toy(nn.Module):
        def __init__(self,mods):
            super().__init__()
            self.shared = nn.Linear(2,2)
            self.transport = nn.Module()
            self.transport.quantizer = nn.Linear(2,2)
            for m in mods:
                for prefix in ('encoder_','backbone_','aligner_'):
                    setattr(self,prefix+m,nn.Linear(2,2))
    cfg = config(stage='detector')
    parents = {}
    for i,m in enumerate(MEMBERS):
        model = Toy([m])
        for p in model.parameters():
            p.data.fill_(i+1)
        path = tmp_path/(m+'.pth')
        parent_cfg = config(stage='pretrain_'+m)
        torch.save(dict(schema=SCHEMA,dataset='opv2v',stage='pretrain_'+m,validated=True,model=model.state_dict(),
                        config=parent_cfg,config_hash=fingerprint(parent_cfg)),path)
        parents['pretrain_'+m] = path
    model = Toy(MEMBERS)
    initial_codebook = model.transport.quantizer.weight.detach().clone()
    initialize_parents(model,cfg,parents)
    assert torch.equal(model.shared.weight,torch.ones_like(model.shared.weight))
    for i,m in enumerate(MEMBERS):
        assert torch.all(getattr(model,'encoder_'+m).weight == i+1)
    torch.testing.assert_close(model.transport.quantizer.weight,initial_codebook)
    with pytest.raises(ValueError,match='required parent'):
        initialize_parents(model,cfg,{})
    parents['pretrain_m3'] = tmp_path/'missing.pth'
    with pytest.raises(FileNotFoundError):
        initialize_parents(model,cfg,parents)


@pytest.mark.parametrize('out_of_range',[(),('m2','m3'),('m2',)])
def test_original_gt_cutoff_on_cpu_fixture(out_of_range):
    # Execute the exact copied DOMA __getitem__ with a tiny sensor/postprocessor
    # fixture: verifies control flow and full GT, not sensor decoding or AP.
    source = (ROOT/'opencood/data_utils/datasets/codefilling_infer_dataset.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == '__getitem__')
    namespace = dict(OrderedDict=OrderedDict,np=np,torch=torch,math=math,
        add_noise_data_dict=lambda data,settings:data,
        get_pairwise_transformation=lambda data,n,proj:np.tile(np.eye(4),(n,n,1,1)),
        merge_features_to_dict=lambda data,**kw:{})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'DOMA_gt_contract','exec'),namespace)
    class Fixture:
        __getitem__ = namespace['__getitem__']
        def __init__(self,n):
            self.use_cav,self.max_cav,self.proj_first = n,5,False
            self.visualize,self.kd_flag,self.box_align,self.supervise_single,self.heterogeneous = False,False,False,True,True
            self.params = dict(noise_setting={},comm_range=70,fusion=dict(dataset='opv2v'),postprocess=dict(max_num=100))
            self.modality_name_list = list(MEMBERS)
            self.sensor_type_dict = dict(m1='lidar',m2='camera',m3='lidar',m4='camera')
            self.anchor_box = np.zeros((1,1,2,7))
            self.post_processor = SimpleNamespace(collate_batch=lambda data:{},generate_label=lambda **kw:{'gt':int(kw['mask'].sum())})
            self.processed = []
        def retrieve_base_data(self,idx):
            return OrderedDict((str(101+i),dict(ego=i==0,modality_name=m,params=dict(
                lidar_pose=[100 if m in out_of_range else i,0,0,0,0,0],
                lidar_pose_clean=[100 if m in out_of_range else i,0,0,0,0,0]))) for i,m in enumerate(MEMBERS))
        def generate_object_center_lidar(self,cavs,pose):
            i = int(cavs[0]['params']['lidar_pose'][0])
            return np.array([[i,0,0,1,1,1,0]]),np.ones(1),[str(i)]
        generate_object_center_camera = generate_object_center_lidar
        def get_item_single_car(self,cav,ego):
            m = cav['modality_name']
            self.processed.append(m)
            return {('processed_features_' if m in ('m1','m3') else 'image_inputs_')+m:{},
                    'single_label_dict':{},'single_object_bbx_center':np.zeros((1,7)), 'single_object_bbx_mask':np.ones(1)}
    gt = []
    available = [(str(101+i),m) for i,m in enumerate(MEMBERS) if m not in out_of_range]
    for n in (1,2,3,4):
        dataset = Fixture(n)
        result = dataset[0]['ego']
        assert dataset.processed == [m for _,m in available[:n]]
        assert result['agent_modality_list'] == dataset.processed
        assert result['cav_num'] == min(n,len(available))
        assert result['label_dict']['gt'] == len(available)
        assert result['cav_id_list'] == [cav for cav,_ in available]
        validate_input_members(result['cav_id_list'],result['cav_id_list'][:result['cav_num']],
                               result['agent_modality_list'],[m for _,m in available],n)
        gt.append(result['object_bbx_center'])
    assert all(np.array_equal(gt[0],x) for x in gt[1:])


def test_native_maxfusion_ast_and_infer_source_provenance():
    def symbols(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('regroup','MaxFusion')}
    assert symbols(ROOT/'opencood/models/fuse_modules/codefilling_max.py') == symbols(ROOT/'opencood/models/fuse_modules/fusion_in_one.py')


def test_missing_real_data_is_not_reported_pass():
    # Local checkout deliberately has no bundled dataset.
    if not (ROOT/'dataset/OPV2V/train').exists():
        with pytest.raises(FileNotFoundError,match='SERVER_VALIDATION_REQUIRED'):
            audit_data('opv2v')


def test_pcd_rgb_and_intensity_branches_without_sensor_dependencies():
    source = (ROOT/'opencood/utils/pcd_utils.py').read_text(encoding='utf-8')
    method = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'pcd_to_np')
    xyz = np.array([[1.,2.,3.],[4.,5.,6.]])
    colors = np.array([[.2,.2,.2],[.8,.8,.8]])
    structured = np.array([(17.,),(23.,)],dtype=[('intensity','f4')])
    namespace = dict(np=np,o3d=SimpleNamespace(io=SimpleNamespace(read_point_cloud=lambda path:SimpleNamespace(points=xyz,colors=colors))),
                     pypcd=SimpleNamespace(PointCloud=SimpleNamespace(from_path=lambda path:SimpleNamespace(pc_data=structured))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'pcd_reader_contract','exec'),namespace)
    rgb = namespace['pcd_to_np']('fixture.pcd')
    np.testing.assert_array_equal(rgb[:,3],np.array([.2,.8],dtype=np.float32))
    colors = np.empty((0,3))
    intensity = namespace['pcd_to_np']('fixture.pcd')
    np.testing.assert_array_equal(intensity[:,3],np.array([17.,23.],dtype=np.float32))
    assert intensity.dtype == np.float32


def test_base_retrieval_propagates_modality_and_rgb_only_hdf5(tmp_path):
    source = (ROOT/'opencood/data_utils/datasets/basedataset/opv2v_basedataset.py').read_text(encoding='utf-8')
    method = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name == 'retrieve_base_data')
    # Exercise the actual loader control flow with a fixture HDF5 facade that
    # deliberately has RGB keys only. Accessing depth would raise KeyError.
    meta = tmp_path/'0001.yaml'
    meta.write_text('{}')
    class RGBBundle:
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def __getitem__(self,key):
            if key not in ['camera'+str(i) for i in range(4)]:
                raise KeyError(key)
            return np.zeros((2,2,3),dtype=np.uint8)
    hdf = tmp_path/'0001_imgs.hdf5'
    hdf.write_bytes(b'fixture')
    namespace = dict(OrderedDict=OrderedDict,os=__import__('os'),json=json,
                     load_yaml=lambda path:{'lidar_pose':[0]*6},h5py=SimpleNamespace(File=lambda *a,**k:RGBBundle()),
                     Image=SimpleNamespace(fromarray=lambda x:x))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'base_retrieval_contract','exec'),namespace)
    frame = dict(yaml=str(meta),modality_name='m3',cameras=[str(tmp_path/'0001_camera0.png')])
    dataset = SimpleNamespace(len_record=[1],scenario_database={0:OrderedDict([('1',{'ego':True,'0001':frame})])},
        return_timestamp_key=lambda data,index:'0001',tau=0,heterogeneous=True,load_camera_file=True,
        load_depth_file=False,load_lidar_file=False,visualize=False,add_data_extension=[])
    data = namespace['retrieve_base_data'](dataset,0)
    assert data['1']['modality_name'] == 'm3'
    assert len(data['1']['camera_data']) == 4 and data['1']['depth_data'] == []


def test_camera_classes_match_recorded_doma_source():
    evidence = json.loads((ROOT/'docs/codefilling_sources/camera_source.json').read_text())
    # Direct live source comparison if the checked-out evidence origin remains
    # available; the portable static gate uses the recorded artifact digest.
    origin = Path(evidence['origin'])
    if origin.is_file():
        def classes(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n,ast.ClassDef) and n.name in evidence['classes']}
        assert classes(origin) == classes(ROOT/'opencood/models/sub_modules/codefilling_camera.py')


@pytest.mark.parametrize('ds',['opv2v','v2xset'])
def test_every_yaml_builds_its_real_cpu_core(ds):
    for path in sorted((ROOT/'opencood/hypes_yaml'/ds/'doma_codefilling').glob('*.yaml')):
        cfg = load_config(path)
        model = CodeFillingCore(cfg['model']['args']).eval()
        model.configure_stage()
        result = model.forward_features(torch.randn(2,64,8,8),torch.tensor([2]),identity(2))
        if path.stem != 'codebook':
            assert result['cls_preds'].shape == (1,2,8,8)
        assert torch.isfinite(result['codebook_loss'])


def test_original_ap_global_score_sorting_contract():
    tree = ast.parse((ROOT/'opencood/utils/eval_utils.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('voc_ap','calculate_ap')]
    namespace = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'native_ap_contract','exec'),namespace)
    stats = {t:dict(tp=[0,1],fp=[1,0],score=[.1,.9],gt=1) for t in (.3,.5,.7)}
    for threshold in stats:
        assert namespace['calculate_ap'](stats,threshold)[0] == pytest.approx(1.)


@pytest.mark.parametrize('probabilities,expected', [
    ([.5,.5], [0]),          # crossing member itself is excluded: 1 is not < 1
    ([.1,.8,.6,.3], [1,0,0]),
    ([.1,.6,.2,.05], [1,1,1]),
    ([.8,.3,.2], [0,0]),
])
def test_default_sum_solver_threshold_zero_is_not_dense(probabilities, expected):
    count = len(probabilities)
    logits = torch.logit(torch.tensor(probabilities))[:,None,None,None].expand(count,2,4,4)
    native = Communication(dict(thre=0.,solver=dict(thre=1.,method='sum')))
    _,masks,_ = native([logits],torch.tensor([count]),identity(count))
    for index, selected in enumerate(expected,1):
        assert torch.all(masks[index] == selected)
    _,dense,_ = Communication(dict(thre=0.))([logits],torch.tensor([count]),identity(count))
    assert torch.all(dense == 1)
    # Float32 sigmoid can underflow to 0: threshold=0 still rejects that value.
    extreme = logits.clone()
    extreme[1] = -1000
    _,mask,_ = Communication(dict(thre=0.))([extreme],torch.tensor([count]),identity(count))
    assert torch.all(mask[1] == 0)


@pytest.mark.parametrize('case',['dense','suppressed','warped'])
def test_inference_messages_match_actual_multiscale_forward(case):
    seed_all(303)
    args = config()['model']['args']
    args['fusion_backbone']['layer_nums'] = [1,1,1]
    model = CodeFillingCore(args).eval()
    model.configure_stage()
    with torch.no_grad():
        model.cls_head_single.weight.zero_()
        model.cls_head_single.weight[:,0,0,0] = 1
        model.cls_head_single.bias.zero_()
    probabilities = [.5]*4 if case == 'suppressed' else ([.1]*4 if case == 'dense' else [.1,.8,.6,.3])
    original = torch.randn(4,64,8,8)
    original[:,0] = torch.logit(torch.tensor(probabilities))[:,None,None]
    transforms = identity(4)
    if case == 'warped':
        transforms[0,0,1,0,2] = .125
        transforms[0,1,0,0,2] = -.125
    captured, handles = {}, []
    handles.append(model.transport.quantizer.register_forward_pre_hook(
        lambda module,values: captured.update(quantizer_input=values[0].clone())))
    handles.append(model.transport.communication.register_forward_hook(
        lambda module,values,result: captured.update(confidence=values[0][0].clone(),mask=result[1].clone())))
    handles.append(model.transport.register_forward_hook(
        lambda module,values,result: captured.update(level0=result[0].clone())))
    for level in (1,2):
        handles.append(getattr(model.backbone.resnet,'layer'+str(level)).register_forward_hook(
            lambda module,values,result,level=level: captured.update({f'level{level}':result.clone()})))
    for level,module in enumerate(model.fusion_net):
        handles.append(module.register_forward_pre_hook(
            lambda module,values,level=level: captured.update({f'message{level}':values[0].clone()})))
        handles.append(module.register_forward_hook(
            lambda module,values,result,level=level: captured.update({f'fused{level}':result.clone()})))
    try:
        with torch.no_grad():
            result = model.forward_features(original,torch.tensor([4]),transforms)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(captured['quantizer_input'],original.permute(0,2,3,1).reshape(-1,64),rtol=0,atol=0)
    # Confidence is computed from original member features, before reconstruction.
    torch.testing.assert_close(captured['confidence'],model.cls_head_single(original),rtol=0,atol=0)
    torch.testing.assert_close(captured['level0'][0],original[0],rtol=0,atol=0)
    decoded = decode_sampled_codes(model.transport.quantizer,model.transport.last_codes).reshape(4,8,8,64).permute(0,3,1,2)
    torch.testing.assert_close(decoded[1:],captured['level0'][1:],rtol=1e-6,atol=1e-6)
    masks = captured['mask']
    for level in (0,1,2):
        torch.testing.assert_close(captured[f'message{level}'],captured[f'level{level}']*masks,rtol=0,atol=0)
        for peer,row in enumerate(result['communication']['messages'],1):
            cells = int(torch.count_nonzero(masks[peer]))
            if level == 0:
                assert row['selected_cells'] == cells
                support = masks[peer,0] != 0
                actual_codes = torch.cat([code[peer][support].flatten() for code in model.transport.last_codes])
                payload = pack_indices(actual_codes.tolist(),args['codebook']['dict_size'])
                assert row['index_bytes'] == len(payload)
            else:
                vector = captured[f'level{level}'][peer]
                assert row['coarse_levels'][level-1]['bytes'] == cells*vector.shape[0]*vector.element_size()
        if level < 2:
            masks = torch.nn.functional.max_pool2d(masks,2)
    if case == 'suppressed':
        assert result['communication']['selected_cells'] == 0
        assert result['communication']['control_bytes'] > 0
        # Native MaxFusion still includes the masked zero peer tensors. They
        # clamp negative ego values; do not silently drop peers to force CAV1 parity.
        assert torch.any(captured['level0'][0] < 0)
        for level in (0,1,2):
            ego_feature = captured[f'level{level}'][:1]
            from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
            warped_ego = warp_affine_simple(ego_feature,transforms[0,0,:1],ego_feature.shape[-2:])
            torch.testing.assert_close(captured[f'fused{level}'],warped_ego.clamp_min(0),rtol=0,atol=0)


@pytest.mark.parametrize('count',[1,4])
def test_native_gumbel_eval_repeatability_requires_rng_reset(count):
    seed_all(303)
    model = CodeFillingCore(core_args('infer')).eval()
    model.configure_stage()
    with torch.no_grad():
        model.cls_head_single.weight.zero_()
        model.cls_head_single.bias.fill_(torch.logit(torch.tensor(.1)).item())
    x = torch.full((count,64,8,8),-10.)
    def run():
        with torch.no_grad():
            output = model.forward_features(x,torch.tensor([count]),identity(count))
        return output['cls_preds'].clone(), [code.clone() for code in model.transport.last_codes]
    seed_all(303)
    initial = torch.get_rng_state().clone()
    first,first_codes = run()
    assert not torch.equal(initial,torch.get_rng_state())
    second,second_codes = run()
    assert any(not torch.equal(x,y) for x,y in zip(first_codes,second_codes))
    if count == 1:
        torch.testing.assert_close(first,second,rtol=0,atol=0)  # ego bypass
    else:
        assert not torch.equal(first,second)
    seed_all(303)
    repeated,repeated_codes = run()
    torch.testing.assert_close(first,repeated,rtol=0,atol=0)
    assert all(torch.equal(x,y) for x,y in zip(first_codes,repeated_codes))


def test_fresh_import_global_module_consumes_rng_but_seed_after_import_is_stable():
    import subprocess
    import sys
    script = '''
import sys,torch,hashlib,json
torch.set_num_threads(1)
torch.manual_seed(303)
before=torch.get_rng_state().clone()
if sys.argv[1]=='early':
 from opencood.models.sub_modules import codebook
 changed=not torch.equal(before,torch.get_rng_state())
from opencood.models.codefilling_doma import CodeFillingCore
from opencood.utils.codefilling_state import seed_all
from opencood.utils.codefilling_protocol import load_config
seed_all(303)
args=load_config('opencood/hypes_yaml/opv2v/doma_codefilling/infer.yaml')['model']['args']
model=CodeFillingCore(args)
h=hashlib.sha256()
for tensor in model.state_dict().values(): h.update(tensor.detach().numpy().tobytes())
print(json.dumps({'hash':h.hexdigest(),'global_consumed_rng':changed if sys.argv[1]=='early' else None}))
'''
    early = json.loads(subprocess.check_output([sys.executable,'-B','-c',script,'early'],cwd=ROOT,text=True))
    late = json.loads(subprocess.check_output([sys.executable,'-B','-c',script,'late'],cwd=ROOT,text=True))
    assert early['global_consumed_rng'] is True
    assert early['hash'] == late['hash']


def test_sampling_seed_starts_after_loader_setup():
    from opencood.tools.inference_codefilling_doma import seeded_evaluation_batches
    class Loader:
        def __iter__(self):
            torch.rand(37)  # native DataLoader iterator setup may consume RNG
            return iter([0])
    seed_all(303)
    expected = torch.rand(5)
    for _ in seeded_evaluation_batches(Loader(),303):
        actual = torch.rand(5)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('dataset,folder',[('opv2v','OPV2V'),('v2xset','V2XSET')])
def test_fixed_stage_output_directories_and_editable_budget(dataset,folder,tmp_path):
    for stage in ('pretrain_m1','pretrain_m2','pretrain_m3','pretrain_m4','detector','codebook','finetune','infer'):
        cfg = config(dataset,stage)
        expected = ROOT/'opencood/logs'/('CodeFilling_'+folder+'_SEED303')/stage
        assert experiment_output(cfg) == expected
        assert experiment_output(cfg,str(expected)) == expected
        with pytest.raises(ValueError,match='formal stage output'):
            experiment_output(cfg,tmp_path/'timestamp_experiment')
    cfg = config(dataset,'pretrain_m1')
    cfg['train_params']['batch_size'],cfg['train_params']['epoches'] = 2,17
    validate_config(cfg)  # budgets remain editable; this is not formal approval


def test_cross_cav_actual_frame_gt_prefix_and_message_identity():
    def row(count):
        realized = min(count,3)
        return dict(requested_cavs=count,realized_cavs=realized,all_cav_ids=['1','2','-1'],
            input_cav_ids=['1','2','-1'][:realized],modalities=list(MEMBERS[:realized]),gt_hash='fixture_gt',sample_idx=0,
            cav_modalities=list(MEMBERS[:3]),
            sample_key=dict(scene='fixture_scene',timestamp='0001',sample_idx=0),
            communication=dict(peers=realized-1,messages=[dict(batch=0,sender=a) for a in range(1,realized)]))
    baseline = validate_evaluation_frame(row(1))
    for count in (2,3,4):
        item = row(count)
        assert validate_evaluation_frame(item) == baseline
        assert item['communication']['messages'][0]['cav_id'] == '2'
        assert item['communication']['messages'][0]['modality'] == 'm2'
    for field,value in [('gt_hash','other'),('sample_key',dict(scene='other_scene',timestamp='0001',sample_idx=0))]:
        item = row(4)
        item[field] = value
        assert validate_evaluation_frame(item) != baseline
    for field,value in [('input_cav_ids',['1','-1','2']),('modalities',['m1','m3','m2']),('realized_cavs',2)]:
        item = row(4)
        item[field] = value
        with pytest.raises(ValueError,match='prefix'):
            validate_evaluation_frame(item)
    item = row(4)
    item['communication']['messages'][0]['sender'] = 0
    with pytest.raises(ValueError,match='sender set'):
        validate_evaluation_frame(item)
    fixture = SimpleNamespace(len_record=[2,4],tau=0,scenario_folders=['test/scene_a','test/scene_b'],
                              scenario_database={0:['0001','0002'],1:['0010','0011']},return_timestamp_key=lambda db,i:db[i])
    assert evaluation_sample_key(fixture,2) == dict(scene='scene_b',timestamp='0010',sample_idx=2)


@pytest.mark.parametrize('ids,modalities', [
    (['641','-1'],['m1','m4']),
    (['641','659','-1'],['m1','m3','m4']),
])
def test_filtered_members_preserve_real_ids_modalities_and_messages(ids,modalities):
    original_ids,original_modalities = ids[:],modalities[:]
    reference = None
    for requested in (1,2,3,4):
        realized = min(requested,len(ids))
        actual_mods = modalities[:realized]
        validate_input_members(ids,ids[:realized],actual_mods,modalities,requested)
        ledger = communication_ledger(torch.ones(realized,1,4,4),[realized],dict(seg_num=2,dict_size=64,r=3))
        row = dict(requested_cavs=requested,realized_cavs=realized,all_cav_ids=ids[:],
            input_cav_ids=ids[:realized],modalities=actual_mods,cav_modalities=modalities[:],
            gt_hash='same_range_filtered_gt',sample_idx=7,
            sample_key=dict(scene='real_id_fixture',timestamp='000123',sample_idx=7),communication=ledger)
        identity_hash = validate_evaluation_frame(row)
        reference = identity_hash if reference is None else reference
        assert identity_hash == reference
        assert row['input_cav_ids'] == original_ids[:realized]
        assert row['modalities'] == original_modalities[:realized]
        for index,message in enumerate(row['communication']['messages'],1):
            assert message['cav_id'] == original_ids[index]
            assert message['modality'] == original_modalities[index]
        assert row['communication']['selected_cells'] == (realized-1)*16
        assert row['communication']['total_bytes'] >= 0
    assert ids == original_ids and modalities == original_modalities
    # Filling a missing slot with m2 would silently relabel a valid m4/m3 CAV.
    with pytest.raises(ValueError):
        validate_input_members(ids,ids[:2],['m1','m2'],modalities,2)
    with pytest.raises(ValueError):
        validate_input_members(ids,ids[:],modalities,['m1','m2'] if len(ids)==2 else ['m1','m2','m4'],4)
    # A filtered subset still requires the actual m1 ego in the first position.
    with pytest.raises(ValueError):
        validate_input_members(ids,ids[:],['m4']+modalities[1:],['m4']+modalities[1:],4)


def _review_native_evaluation(path=None):
    """Run only native AP functions; no sensor/CUDA module imports or inference."""
    import os
    path = path or ROOT/'opencood/utils/eval_utils.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and
                 n.name in ('voc_ap','calculate_ap','eval_final_results')]
    saved = {}
    namespace = dict(np=np,os=os,yaml_utils=SimpleNamespace(
        save_yaml=lambda values,path:saved.update(values=copy.deepcopy(values),path=str(path))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(path),'exec'),namespace)
    return SimpleNamespace(**namespace),saved


def _review_ap_stats():
    return {threshold:dict(tp=[1,0,1],fp=[0,1,0],score=[.7,.9,.2],gt=2) for threshold in (.3,.5,.7)}


def test_final_eval_yaml_matches_native_doma_fields_and_full_curves():
    from opencood.tools.inference_codefilling_doma import EVAL_FIELDS,evaluation_values,complete_evaluation
    native,saved = _review_native_evaluation()
    stats = _review_ap_stats()
    actual = evaluation_values(stats,native.calculate_ap)
    native.eval_final_results(stats,'fixture','intermediate_use_cav1')
    assert actual == saved['values']
    assert tuple(actual) == EVAL_FIELDS
    assert saved['path'].endswith('eval_intermediate_use_cav1.yaml')
    doma = ROOT.parents[1]/'DOMA/opencood/utils/eval_utils.py'
    if doma.is_file():
        reference,reference_saved = _review_native_evaluation(doma)
        reference.eval_final_results(stats,'fixture','intermediate_use_cav1')
        assert actual == reference_saved['values']
    assert complete_evaluation(actual)
    for broken in ({},dict(actual,ap30=float('nan')),dict(actual,mrec_50=[]),
                   dict(actual,mpre_70=[0.]),dict(actual,ap_50=1.1)):
        assert not complete_evaluation(broken)


def test_final_infer_history_is_backed_up_without_deleting_any_file(tmp_path):
    from opencood.tools.inference_codefilling_doma import prepare_output
    output = tmp_path/'infer'
    (output/'partial').mkdir(parents=True)
    (output/'infer.log').write_bytes(b'historical log\n')
    (output/'partial'/'old.txt').write_bytes(b'incomplete result retained\n')
    before = {str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    backup = prepare_output(output)
    assert backup.parent == output.parent and backup != output
    assert output.is_dir() and not list(output.iterdir())
    assert {str(p.relative_to(backup)):p.read_bytes() for p in backup.rglob('*') if p.is_file()} == before
    assert prepare_output(output) is None
    (output/'new_partial.txt').write_text('another interrupted result')
    second = prepare_output(output)
    assert second != backup and (second/'new_partial.txt').read_text() == 'another interrupted result'
    assert (backup/'infer.log').read_bytes() == b'historical log\n'


def test_completed_log_and_eval_yaml_reuse_requires_matching_context_and_sha(tmp_path):
    import yaml
    from opencood.tools.inference_codefilling_doma import evaluation_values,reusable_cav1,signature
    from opencood.utils.codefilling_protocol import digest
    native,_ = _review_native_evaluation()
    values = evaluation_values(_review_ap_stats(),native.calculate_ap)
    output = tmp_path/'infer'
    output.mkdir()
    path = output/'eval_intermediate_use_cav1.yaml'
    path.write_text(yaml.safe_dump(values,sort_keys=False))
    context = dict(checkpoint_sha256='same-checkpoint',config_sha256='same-config',assignment_sha256='same-assignment')
    checked = signature(['frame-0-with-gt','frame-1-with-gt'])
    completed = dict(requested_cavs=1,**checked,eval_file=path.name,eval_sha256=digest(path))
    (output/'infer.log').write_text('CODEFILLING_RUN '+json.dumps(context)+'\nCAV_COMPLETE '+json.dumps(completed)+'\n')
    cache = reusable_cav1(output,config('v2xset'),context,{},native.calculate_ap)
    assert cache['values'] == values and cache['signature'] == checked
    with (output/'infer.log').open('a') as stream:
        stream.write('CAV_COMPLETE {"requested_cavs":2,')
    assert reusable_cav1(output,config('v2xset'),context,{},native.calculate_ap)['values'] == values
    assert reusable_cav1(output,config('v2xset'),dict(context,checkpoint_sha256='other-checkpoint'),{},native.calculate_ap) is None
    path.write_text(yaml.safe_dump(dict(values,ap30=.123),sort_keys=False))
    assert reusable_cav1(output,config('v2xset'),context,{},native.calculate_ap) is None
    # A partial completion record must not be accepted as a finished CAV1 AP.
    (output/'infer.log').write_text('CODEFILLING_RUN '+json.dumps(context)+'\nCAV_COMPLETE {')
    assert reusable_cav1(output,config('v2xset'),context,{},native.calculate_ap) is None


@pytest.mark.parametrize('saved_format',['raw','yaml'])
def test_legacy_completed_cav1_raw_ap_and_frames_can_migrate_without_forward(tmp_path,saved_format):
    import yaml
    from opencood.tools.inference_codefilling_doma import reusable_cav1,evaluation_values,signature
    cfg = config('v2xset')
    assignment = {'fixture_scene':{'641':'m1','-1':'m4'}}
    context = dict(checkpoint_sha256='legacy-checkpoint',config_sha256=fingerprint(cfg))
    output = tmp_path/'infer'
    output.mkdir()
    (output/'run.json').write_text(json.dumps(dict(checkpoint_sha256=context['checkpoint_sha256'])))
    (output/'config.yaml').write_text(yaml.safe_dump(cfg))
    rows = [dict(index=i,sample_idx=i,sample_key=dict(scene='fixture_scene',timestamp=f'{i:04}',sample_idx=i),
                 requested_cavs=1,realized_cavs=1,all_cav_ids=['641','-1'],input_cav_ids=['641'],modalities=['m1'],
                 gt_hash='frame-'+str(i)+'-GT',communication=dict(peers=0,messages=[],total_bytes=0)) for i in range(2)]
    frames = output/'cav1_frames.jsonl'
    frames.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    native,_ = _review_native_evaluation()
    raw = output/('cav1_raw_ap.json' if saved_format == 'raw' else 'eval_intermediate_use_cav1.yaml')
    raw.write_text(json.dumps(_review_ap_stats() if saved_format == 'raw' else
                              evaluation_values(_review_ap_stats(),native.calculate_ap)))
    cache = reusable_cav1(output,cfg,context,assignment,native.calculate_ap)
    identities = []
    for row in copy.deepcopy(rows):
        row['cav_modalities'] = ['m1','m4']
        identities.append(validate_evaluation_frame(row))
    assert cache['signature'] == signature(identities)
    assert cache['values'] == evaluation_values(_review_ap_stats(),native.calculate_ap)
    assert cache['source'] == raw.name
    frames.write_text(json.dumps(rows[0])+'\n{')
    assert reusable_cav1(output,cfg,context,assignment,native.calculate_ap) is None
    frames.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    invalid = _review_ap_stats() if saved_format == 'raw' else {}
    if saved_format == 'raw':
        invalid[.5]['score'] = []
    raw.write_text(json.dumps(invalid))
    assert reusable_cav1(output,cfg,context,assignment,native.calculate_ap) is None


class _ReviewInferenceDataset(torch.utils.data.Dataset):
    """A two-frame GT fixture, never an actual perception Dataset or model."""
    def __init__(self,count):
        self.count = count
        self.len_record,self.tau = [2],0
        self.scenario_folders = ['fixture/fixture_scene']
        self.scenario_database = {0:['0000','0001']}
        self.post_processor = SimpleNamespace(generate_gt_bbx=self.gt)

    @staticmethod
    def return_timestamp_key(database,index):
        return database[index]

    @staticmethod
    def gt(batch):
        return torch.full((1,8,3),float(batch['ego']['sample_idx']))

    def __len__(self):
        return 2

    def __getitem__(self,index):
        return {'ego':dict(sample_idx=index,cav_id_list=['641','-1'],agent_modality_list=['m1','m4'][:self.count])}

    @staticmethod
    def collate_batch_test(batch):
        return batch[0]

    def post_process(self,batch,prediction):
        return None,None,self.gt(batch)


def _review_fixture_eval_utils():
    native,_ = _review_native_evaluation()
    def add_result(boxes,scores,gt,stats,threshold):
        stats[threshold]['gt'] += len(gt)
        stats[threshold]['tp'].append(1)
        stats[threshold]['fp'].append(0)
        stats[threshold]['score'].append(.8)
    return SimpleNamespace(calculate_ap=native.calculate_ap,caluclate_tp_fp=add_result)


def test_reusing_cav1_builds_gt_identity_without_any_model_forward(capsys):
    from opencood.tools.inference_codefilling_doma import evaluate_group,evaluation_values,signature,tensor_hash
    cfg = config('v2xset')
    assignment = {'fixture_scene':{'641':'m1','-1':'m4'}}
    dataset = _ReviewInferenceDataset(1)
    identities = []
    for index in range(2):
        batch = dataset[index]
        row = dict(sample_idx=index,sample_key=evaluation_sample_key(dataset,index),all_cav_ids=['641','-1'],
                   input_cav_ids=['641'],modalities=['m1'],cav_modalities=['m1','m4'],requested_cavs=1,realized_cavs=1,
                   gt_hash=tensor_hash(dataset.gt(batch)),communication=dict(peers=0,messages=[],total_bytes=0))
        identities.append(validate_evaluation_frame(row))
    native = _review_fixture_eval_utils()
    values = evaluation_values(_review_ap_stats(),native.calculate_ap)
    cache = dict(values=values,signature=signature(identities))
    def no_model_forward(batch):
        raise AssertionError('a reusable CAV1 must not call its model')
    result,summary = evaluate_group(cfg,1,no_model_forward,'cpu',assignment,
        lambda *args,**kwargs:dataset,native,reuse=cache)
    assert result == values and summary['reused'] is True
    assert summary['mean_bytes'] == 0 and summary['frames'] == 2
    assert 'GT-check/reuse 2/2' in capsys.readouterr().out


def test_fixture_main_writes_only_log_and_four_eval_yamls_and_reuses_cav1(tmp_path,monkeypatch):
    import sys
    import yaml
    from opencood.tools import inference_codefilling_doma as entry
    cfg = config('v2xset')
    assignment = tmp_path/'assignment.json'
    assignment.write_text(json.dumps({'fixture_scene':{'641':'m1','-1':'m4'}}))
    cfg['heter']['assignment_path'] = str(assignment)
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path/'fixture_checkpoint.bin'
    checkpoint.write_bytes(b'CPU contract fixture; no trained weights')
    output = tmp_path/'infer'
    calls = []
    class FixtureModel:
        def __init__(self,args): pass
        def to(self,device): return self
        def eval(self): return self
        def load_state_dict(self,state,strict): assert state == {} and strict
        def __call__(self,batch):
            mods = batch['agent_modality_list']
            calls.append(mods[:])
            peers = len(mods)-1
            return {'communication':dict(peers=peers,messages=[dict(batch=0,sender=i) for i in range(1,peers+1)],total_bytes=peers*100)}
    monkeypatch.setitem(sys.modules,'opencood.models.codefilling_doma',SimpleNamespace(CodefillingDoma=FixtureModel))
    monkeypatch.setitem(sys.modules,'opencood.data_utils.codefilling_dataset',SimpleNamespace(
        build_codefilling_dataset=lambda current,**kw:_ReviewInferenceDataset(current['use_cav'])))
    import opencood.utils
    fake_eval = _review_fixture_eval_utils()
    monkeypatch.setitem(sys.modules,'opencood.utils.eval_utils',fake_eval)
    monkeypatch.setattr(opencood.utils,'eval_utils',fake_eval,raising=False)
    monkeypatch.setattr(entry,'read_checkpoint',lambda *args,**kw:dict(model={}))
    monkeypatch.setattr(entry,'experiment_output',lambda *args,**kw:output)
    monkeypatch.setattr(sys,'argv',['inference_fixture','--config',str(config_path),'--checkpoint',str(checkpoint),'--device','cpu'])
    entry.main()
    expected = {'infer.log'}|{f'eval_intermediate_use_cav{n}.yaml' for n in (1,2,3,4)}
    assert {p.name for p in output.iterdir()} == expected
    assert len(calls) == 8
    before = {p.name:p.read_bytes() for p in output.iterdir()}
    for n in (1,2,3,4):
        values = yaml.safe_load(before[f'eval_intermediate_use_cav{n}.yaml'])
        assert tuple(values) == entry.EVAL_FIELDS and values['ap30'] == 1.
    entry.main()
    assert len(calls) == 14  # second run calls CAV2-4 only; CAV1 is GT-only
    assert {p.name for p in output.iterdir()} == expected
    assert (output/'eval_intermediate_use_cav1.yaml').read_bytes() == before['eval_intermediate_use_cav1.yaml']
    backups = list(tmp_path.glob('infer_backup_*'))
    assert len(backups) == 1
    assert {p.name:p.read_bytes() for p in backups[0].iterdir()} == before
    assert 'CAV1 GT-check/reuse' in (output/'infer.log').read_text()


def test_dataset_wrapper_accepts_actual_m1_m4_item_without_relabeling():
    source = (ROOT/'opencood/data_utils/codefilling_dataset.py').read_text(encoding='utf-8')
    method = next(node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef) and node.name == '__getitem__')
    item = {'ego':dict(agent_modality_list=['m1','m4'],cav_id_list=['641','-1'])}
    class Parent:
        def __getitem__(self,index):
            return item
    wrapper = ast.ClassDef(name='ProtocolDataset',bases=[ast.Name(id='parent',ctx=ast.Load())],keywords=[],
                           body=[method],decorator_list=[],type_params=[])
    namespace = dict(parent=Parent,inference=True,cfg=config('v2xset'),
                     evaluation_sample_key=evaluation_sample_key,validate_input_members=validate_input_members)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'actual_protocol_wrapper','exec'),namespace)
    dataset = namespace['ProtocolDataset']()
    dataset.use_cav,dataset.len_record,dataset.tau = 4,[1],0
    dataset.scenario_folders = ['test/fixture_scene']
    dataset.scenario_database = {0:['0001']}
    dataset.return_timestamp_key = lambda database,index:database[index]
    dataset.modality_assignment = {'fixture_scene':{'641':'m1','-1':'m4'}}
    assert dataset[0] is item
    assert item['ego']['cav_id_list'] == ['641','-1']
    assert item['ego']['agent_modality_list'] == ['m1','m4']
    dataset.modality_assignment['fixture_scene']['-1'] = 'm2'
    with pytest.raises(ValueError,match='mapping'):
        dataset[0]


def test_doma_evaluation_protocol_off_preserves_config_identity_and_fingerprint():
    from opencood.tools import inference_codefilling_doma as entry
    cfg = config('v2xset')
    original = copy.deepcopy(cfg)
    original_hash = fingerprint(cfg)
    assert entry.apply_doma_evaluation_protocol(cfg) is cfg
    assert entry.apply_doma_evaluation_protocol(cfg,enabled=False) is cfg
    assert cfg == original and fingerprint(cfg) == original_hash
    assert 'evaluation_protocol' not in cfg['codefilling_protocol']


@pytest.mark.parametrize('dataset',['opv2v','v2xset'])
def test_doma_evaluation_protocol_on_synchronizes_ranges_without_mutating_cameras(dataset):
    from opencood.tools import inference_codefilling_doma as entry
    cfg = config(dataset)
    original = copy.deepcopy(cfg)
    enabled = entry.apply_doma_evaluation_protocol(cfg,enabled=True)
    expected = [-204.8,-102.4,-3,204.8,102.4,1]
    assert enabled is not cfg and cfg == original
    assert list(entry.DOMA_EVALUATION_BOUNDS) == expected
    assert entry.DOMA_COMMUNICATION_RANGE == 180
    ranges = [enabled['cav_lidar_range'],enabled['preprocess']['cav_lidar_range'],
        enabled['postprocess']['gt_range'],enabled['postprocess']['anchor_args']['cav_lidar_range'],
        enabled['model']['args']['lidar_range']]
    for member in ('m1','m3'):
        ranges.extend([enabled['heter']['modality_setting'][member]['preprocess']['cav_lidar_range'],
                       enabled['model']['args'][member]['encoder_args']['lidar_range']])
    assert len(ranges) == 9 and all(bounds == expected for bounds in ranges)
    assert enabled['comm_range'] == 180
    assert enabled['codefilling_protocol']['evaluation_protocol'] == dict(
        name='doma',eval_bounds=expected,comm_range=180)
    for member in ('m2','m4'):
        assert enabled['heter']['modality_setting'][member] == original['heter']['modality_setting'][member]
        assert enabled['model']['args'][member]['encoder_args'] == original['model']['args'][member]['encoder_args']
        assert enabled['model']['args'][member]['camera_mask_args']['cav_lidar_range'] == expected
        assert enabled['model']['args'][member]['camera_mask_args']['grid_conf'] == original['model']['args'][member]['camera_mask_args']['grid_conf']
    # Every remaining field must equal the old path: no voxel-size, model,
    # augmentation, seed, assignment or stage/output changes hidden in the switch.
    restored = copy.deepcopy(enabled)
    restored['cav_lidar_range'] = original['cav_lidar_range']
    restored['preprocess']['cav_lidar_range'] = original['preprocess']['cav_lidar_range']
    restored['postprocess']['gt_range'] = original['postprocess']['gt_range']
    restored['postprocess']['anchor_args']['cav_lidar_range'] = original['postprocess']['anchor_args']['cav_lidar_range']
    restored['model']['args']['lidar_range'] = original['model']['args']['lidar_range']
    for member in ('m1','m3'):
        restored['heter']['modality_setting'][member]['preprocess']['cav_lidar_range'] = original['heter']['modality_setting'][member]['preprocess']['cav_lidar_range']
        restored['model']['args'][member]['encoder_args']['lidar_range'] = original['model']['args'][member]['encoder_args']['lidar_range']
    for member in ('m2','m4'):
        restored['model']['args'][member]['camera_mask_args']['cav_lidar_range'] = original['model']['args'][member]['camera_mask_args']['cav_lidar_range']
    restored['comm_range'] = original['comm_range']
    restored['codefilling_protocol'].pop('evaluation_protocol')
    assert restored == original
    assert fingerprint(enabled) != fingerprint(cfg)
    # The marker distinguishes the switch even if an OFF config already has
    # numerically identical ranges and communication distance.
    off_same_numbers = copy.deepcopy(enabled)
    off_same_numbers['codefilling_protocol'].pop('evaluation_protocol')
    assert fingerprint(entry.apply_doma_evaluation_protocol(off_same_numbers)) != fingerprint(enabled)


@pytest.mark.parametrize('dataset',['opv2v','v2xset'])
def test_doma_evaluation_protocol_anchor_dimensions_are_regenerated(dataset):
    from opencood.tools import inference_codefilling_doma as entry
    from opencood.hypes_yaml.yaml_utils import load_general_params
    cfg = config(dataset)
    cfg['postprocess']['anchor_args'].update(W=1,H=1,D=999,vw=123,vh=456,vd=789)
    enabled = entry.apply_doma_evaluation_protocol(cfg,enabled=True)
    parsed = load_general_params(enabled)
    anchors = parsed['postprocess']['anchor_args']
    assert [anchors[key] for key in ('W','H','D')] == [1024,512,1]
    assert [anchors[key] for key in ('vw','vh','vd')] == cfg['preprocess']['args']['voxel_size']
    assert cfg['postprocess']['anchor_args']['W'] == 1
    assert anchors['cav_lidar_range'] == list(entry.DOMA_EVALUATION_BOUNDS)


@pytest.mark.parametrize('changed',['bounds','communication'])
def test_doma_evaluation_protocol_range_and_communication_constants_are_independent(monkeypatch,changed):
    from opencood.tools import inference_codefilling_doma as entry
    cfg = config('v2xset')
    if changed == 'bounds':
        new_bounds = (-102.4,-51.2,-3,102.4,51.2,1)
        monkeypatch.setattr(entry,'DOMA_EVALUATION_BOUNDS',new_bounds)
        enabled = entry.apply_doma_evaluation_protocol(cfg,enabled=True)
        assert enabled['cav_lidar_range'] == list(new_bounds)
        assert enabled['comm_range'] == 180
    else:
        monkeypatch.setattr(entry,'DOMA_COMMUNICATION_RANGE',70)
        enabled = entry.apply_doma_evaluation_protocol(cfg,enabled=True)
        assert enabled['cav_lidar_range'] == [-204.8,-102.4,-3,204.8,102.4,1]
        assert enabled['comm_range'] == 70
    assert enabled['codefilling_protocol']['evaluation_protocol']['eval_bounds'] == enabled['cav_lidar_range']
    assert enabled['codefilling_protocol']['evaluation_protocol']['comm_range'] == enabled['comm_range']


@pytest.mark.parametrize('cache_format',['log','legacy'])
@pytest.mark.parametrize('stored_enabled',[False,True])
def test_cav1_reuse_accepts_same_protocol_and_rejects_cross_protocol(tmp_path,cache_format,stored_enabled):
    import yaml
    from opencood.tools import inference_codefilling_doma as entry
    from opencood.hypes_yaml.yaml_utils import load_general_params
    from opencood.utils.codefilling_protocol import digest
    native,_ = _review_native_evaluation()
    base = config('v2xset')
    stored = load_general_params(entry.apply_doma_evaluation_protocol(copy.deepcopy(base),enabled=stored_enabled))
    other = load_general_params(entry.apply_doma_evaluation_protocol(copy.deepcopy(base),enabled=not stored_enabled))
    assert fingerprint(stored) != fingerprint(other)
    context = dict(checkpoint_sha256='same-finetune-checkpoint',config_sha256=fingerprint(stored),assignment_sha256='same-assignment')
    other_context = dict(context,config_sha256=fingerprint(other))
    assignment = {'fixture_scene':{'641':'m1','-1':'m4'}}
    row = dict(index=0,sample_idx=0,sample_key=dict(scene='fixture_scene',timestamp='0001',sample_idx=0),
               requested_cavs=1,realized_cavs=1,all_cav_ids=['641','-1'],input_cav_ids=['641'],modalities=['m1'],
               cav_modalities=['m1','m4'],gt_hash='same-frame-and-gt',communication=dict(peers=0,messages=[],total_bytes=0))
    checked = entry.signature([validate_evaluation_frame(copy.deepcopy(row))])
    values = entry.evaluation_values(_review_ap_stats(),native.calculate_ap)
    output = tmp_path/'infer'
    output.mkdir()
    path = output/'eval_intermediate_use_cav1.yaml'
    path.write_text(yaml.safe_dump(values,sort_keys=False))
    if cache_format == 'log':
        completed = dict(requested_cavs=1,**checked,eval_file=path.name,eval_sha256=digest(path))
        (output/'infer.log').write_text('CODEFILLING_RUN '+json.dumps(context)+'\nCAV_COMPLETE '+json.dumps(completed)+'\n')
    else:
        (output/'run.json').write_text(json.dumps(dict(checkpoint_sha256=context['checkpoint_sha256'])))
        (output/'config.yaml').write_text(yaml.safe_dump(stored))
        (output/'cav1_frames.jsonl').write_text(json.dumps(row)+'\n')
    reused = entry.reusable_cav1(output,stored,context,assignment,native.calculate_ap)
    assert reused['values'] == values and reused['signature'] == checked
    assert entry.reusable_cav1(output,other,other_context,assignment,native.calculate_ap) is None
    # Keep the protocol name/state unchanged: actual numeric settings must also
    # invalidate cache, rather than relying only on the ON/OFF marker.
    for changed in ('comm_range','cav_lidar_range')+ (('evaluation_protocol',) if stored_enabled else ()):
        altered = copy.deepcopy(stored)
        if changed == 'comm_range':
            altered['comm_range'] += 1
            if stored_enabled:
                altered['codefilling_protocol']['evaluation_protocol']['comm_range'] = altered['comm_range']
        elif changed == 'cav_lidar_range':
            bounds = altered['cav_lidar_range'][:]
            bounds[0] -= 3.2
            bounds[3] += 3.2
            altered['cav_lidar_range'] = bounds[:]
            altered['preprocess']['cav_lidar_range'] = bounds[:]
            altered['postprocess']['gt_range'] = bounds[:]
            altered['postprocess']['anchor_args']['cav_lidar_range'] = bounds[:]
            altered['model']['args']['lidar_range'] = bounds[:]
            for member in ('m1','m3'):
                altered['heter']['modality_setting'][member]['preprocess']['cav_lidar_range'] = bounds[:]
                altered['model']['args'][member]['encoder_args']['lidar_range'] = bounds[:]
            for member in ('m2','m4'):
                altered['model']['args'][member]['camera_mask_args']['cav_lidar_range'] = bounds[:]
            if stored_enabled:
                altered['codefilling_protocol']['evaluation_protocol']['eval_bounds'] = bounds[:]
        else:
            # Same numerical bounds/communication settings, but a legacy/manual
            # config without the explicit protocol marker is a different context.
            altered['codefilling_protocol'].pop('evaluation_protocol')
        altered_context = dict(context,config_sha256=fingerprint(altered))
        assert altered_context['config_sha256'] != context['config_sha256']
        assert entry.reusable_cav1(output,altered,altered_context,assignment,native.calculate_ap) is None
    # Cache inspection must not modify any saved evaluation result.
    assert yaml.safe_load(path.read_text()) == values


def test_doma_evaluation_protocol_cli_is_opt_in_and_applied_before_anchor_and_context():
    import argparse
    source = (ROOT/'opencood/tools/inference_codefilling_doma.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'main')
    parse_index = next(i for i,node in enumerate(main.body) if isinstance(node,ast.Assign) and
                       any(isinstance(t,ast.Name) and t.id == 'opt' for t in node.targets))
    namespace = dict(argparse=argparse,__doc__='static CLI fixture')
    exec(compile(ast.fix_missing_locations(ast.Module(body=main.body[:parse_index],type_ignores=[])),'protocol_cli_fixture','exec'),namespace)
    args = ['--config','fixture/config.yaml','--checkpoint','fixture/checkpoint.pth']
    off = namespace['parser'].parse_args(args)
    on = namespace['parser'].parse_args(args+['--doma-evaluation-protocol'])
    assert off.doma_evaluation_protocol is False and on.doma_evaluation_protocol is True
    assert {key:value for key,value in vars(on).items() if key != 'doma_evaluation_protocol'} == {
        key:value for key,value in vars(off).items() if key != 'doma_evaluation_protocol'}
    helper_index = next(i for i,node in enumerate(main.body) if any(isinstance(n,ast.Call) and
        isinstance(n.func,ast.Name) and n.func.id == 'apply_doma_evaluation_protocol' for n in ast.walk(node)))
    stage_index = next(i for i,node in enumerate(main.body) if isinstance(node,ast.If) and
                       "['stage']" in ast.unparse(node.test))
    anchor_index = next(i for i,node in enumerate(main.body) if any(isinstance(n,ast.Call) and
        isinstance(n.func,ast.Name) and n.func.id == 'load_general_params' for n in ast.walk(node)))
    context_index = next(i for i,node in enumerate(main.body) if any(isinstance(n,ast.Call) and
        isinstance(n.func,ast.Name) and n.func.id == 'fingerprint' for n in ast.walk(node)))
    assert stage_index < helper_index < anchor_index < context_index
