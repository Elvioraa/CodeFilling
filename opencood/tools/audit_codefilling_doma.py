"""Reproducible local source/compile audit and complete unstaged file inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from opencood.utils.codefilling_protocol import ROOT

BASE = '57f83cc'
NATIVE = ['opencood/models/sub_modules/codebook.py','opencood/utils/codebook_utils.py','opencood/models/comm_modules/where2comm.py',
          'opencood/models/fuse_modules/fusion_in_one.py','opencood/models/heter_model_shared_head_solver.py',
          'opencood/models/sub_modules/base_bev_backbone_resnet.py','opencood/models/sub_modules/downsample_conv.py',
          'opencood/loss/point_pillar_depth_loss.py','opencood/loss/point_pillar_loss.py','opencood/utils/eval_utils.py']


def git(*args, root=ROOT):
    # Read-only invocation-local trust; never modify the user's global Git config.
    return subprocess.check_output(['git','-c','safe.directory='+str(root).replace('\\','/'),'-C',str(root),*args])


def purpose(path):
    exact = {
        '.gitignore':'忽略本地数据、权重和 CPU 测试临时产物',
        'opencood/data_utils/datasets/basedataset/opv2v_basedataset.py':'补齐 modality 传播、Windows basename、默认 augmentation 和 RGB-only HDF5',
        'opencood/models/heter_encoders.py':'LSS nonpersistent device buffers；仅新模型 opt-in 使用 DOMA camera contract',
        'opencood/models/sub_modules/torch_transformation_utils.py':'移除未使用的 icecream import，使原生几何 warp 可做 CPU 契约测试',
        'opencood/utils/pcd_utils.py':'保留 RGB intensity，兼容显式 intensity 字段并拒绝无第四特征',
        'opencood/data_utils/datasets/codefilling_infer_dataset.py':'原样复制 DOMA ordered GT-before-cutoff 数据逻辑',
        'opencood/data_utils/codefilling_dataset.py':'独立协议 Dataset/PCD 路径/CAV identity 和统一场景排除',
        'opencood/models/codefilling_doma.py':'独立四成员 CodeFilling 组装、训练冻结与真实 CPU 核心',
        'opencood/models/sub_modules/codefilling_camera.py':'与 DOMA 源类 AST 一致的两个相机实现',
        'opencood/models/sub_modules/codefilling_transport.py':'原生码本/Filling 包装、实际 sampled codes 与多尺度字节账本',
        'opencood/models/fuse_modules/codefilling_max.py':'从原生文件逐字提取的 regroup/MaxFusion',
        'opencood/utils/codefilling_protocol.py':'来源/四模态/范围/服务器 receipt 契约',
        'opencood/utils/codefilling_state.py':'严格父权重所有权、完整 checkpoint 和 epoch 续训',
        'tests/test_codefilling_doma.py':'Windows 静态与 CPU 实际算法/数据流/状态测试'}
    if path in exact:
        return exact[path]
    if path.startswith('opencood/tools/'):
        return '独立 '+Path(path).stem+' 命令入口'
    if '/doma_codefilling/' in path:
        return '独立实验/训练阶段配置，不覆盖官方 YAML'
    if path.startswith('opencood/modality_assign/'):
        return '从 DOMA 原样复制的训练或 ordered assignment'
    if path.startswith('docs/codefilling_sources/'):
        return '协议源 YAML、SHA256、来源路径或 camera evidence'
    return '适配报告、命令模板或本地核验记录'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doma-root', required=True)
    opt = parser.parse_args()
    doma = Path(opt.doma_root)
    audit = dict(initial_head=git('rev-parse',BASE).decode().strip(),initial_status='clean (observed before edits)',
                 newline_policy='normalize CRLF to LF when comparing Git blob to Windows checkout', native_files=[])
    for path in NATIVE:
        old = git('show',BASE+':'+path).replace(b'\r\n',b'\n')
        new = (ROOT/path).read_bytes().replace(b'\r\n',b'\n')
        audit['native_files'].append(dict(path=path,unchanged=old==new,
            base_lf_sha256=hashlib.sha256(old).hexdigest(),current_lf_sha256=hashlib.sha256(new).hexdigest()))
    if not all(item['unchanged'] for item in audit['native_files']):
        raise ValueError('native algorithm file differs from original baseline')
    git('diff','--exit-code',BASE,'--',*NATIVE)
    audit['doma_head'] = git('rev-parse','HEAD',root=doma).decode().strip()
    audit['copied_inference_dataset_exact'] = (ROOT/'opencood/data_utils/datasets/codefilling_infer_dataset.py').read_bytes() == (
        doma/'opencood/data_utils/datasets/heter_infer/intermediate_heter_infer_fusion_dataset.py').read_bytes()
    changes = set(git('ls-files','--modified','--others','--exclude-standard').decode().splitlines())
    audit['python_compiled'] = []
    for path in sorted(changes):
        if path.endswith('.py'):
            compile((ROOT/path).read_text(encoding='utf-8'),path,'exec')
            audit['python_compiled'].append(path)
    (ROOT/'docs/codefilling_native_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    outputs = {'docs/codefilling_native_audit.json','docs/CODEFILLING_CHANGED_FILES.md','docs/codefilling_changed_files.json'}
    changes.update(outputs)
    tracked = set(git('ls-files').decode().splitlines())
    inventory = [dict(path=path,status='modified' if path in tracked else 'added',purpose=purpose(path)) for path in sorted(changes)]
    (ROOT/'docs/codefilling_changed_files.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    markdown = '# CodeFilling 全部修改文件\n\n基线 `'+audit['initial_head']+'`；初始工作区干净。当前未暂存、未提交。\n\n'
    markdown += '总计 '+str(len(inventory))+' 个文件：'+str(sum(row['status']=='modified' for row in inventory))+' 个修改，其余为新增。测试临时文件、数据与权重不计入源码清单。\n\n'
    markdown += '| 文件 | 状态 | 作用 |\n|---|---|---|\n'
    markdown += ''.join('| `'+row['path']+'` | '+row['status']+' | '+row['purpose']+' |\n' for row in inventory)
    (ROOT/'docs/CODEFILLING_CHANGED_FILES.md').write_text(markdown,encoding='utf-8')
    print(json.dumps(dict(native_unchanged=len(NATIVE),python_compiled=len(audit['python_compiled']),changed_files=len(inventory)),indent=2))


if __name__ == '__main__':
    main()
