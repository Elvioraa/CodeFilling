# CodeFilling 全部修改文件

基线 `57f83ccc7d8457da74f06424811c130c3020e19e`；初始工作区干净。当前未暂存、未提交。

总计 60 个文件：5 个修改，其余为新增。测试临时文件、数据与权重不计入源码清单。

| 文件 | 状态 | 作用 |
|---|---|---|
| `.gitignore` | modified | 忽略本地数据、权重和 CPU 测试临时产物 |
| `docs/CODEFILLING_CHANGED_FILES.md` | added | 适配报告、命令模板或本地核验记录 |
| `docs/DOMA_OPV2V_V2XSET_ADAPTATION.md` | added | 适配报告、命令模板或本地核验记录 |
| `docs/codefilling_changed_files.json` | added | 适配报告、命令模板或本地核验记录 |
| `docs/codefilling_data_checks.json` | added | 适配报告、命令模板或本地核验记录 |
| `docs/codefilling_local_checks.json` | added | 适配报告、命令模板或本地核验记录 |
| `docs/codefilling_native_audit.json` | added | 适配报告、命令模板或本地核验记录 |
| `docs/codefilling_sources/camera_source.json` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/opv2v/final.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/opv2v/lock.json` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/opv2v/m1.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/opv2v/m2.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/opv2v/m3.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/opv2v/m4.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/v2xset/final.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/v2xset/lock.json` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/v2xset/m1.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/v2xset/m2.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/v2xset/m3.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_sources/v2xset/m4.yaml` | added | 协议源 YAML、SHA256、来源路径或 camera evidence |
| `docs/codefilling_test_results.txt` | added | 适配报告、命令模板或本地核验记录 |
| `docs/server_settings.example.yaml` | added | 适配报告、命令模板或本地核验记录 |
| `opencood/data_utils/codefilling_dataset.py` | added | 独立协议 Dataset/PCD 路径/CAV identity 和统一场景排除 |
| `opencood/data_utils/datasets/basedataset/opv2v_basedataset.py` | modified | 补齐 modality 传播、Windows basename、默认 augmentation 和 RGB-only HDF5 |
| `opencood/data_utils/datasets/codefilling_infer_dataset.py` | added | 原样复制 DOMA ordered GT-before-cutoff 数据逻辑 |
| `opencood/hypes_yaml/opv2v/doma_codefilling/codebook.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/opv2v/doma_codefilling/detector.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/opv2v/doma_codefilling/finetune.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/opv2v/doma_codefilling/infer.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/opv2v/doma_codefilling/pretrain_m1.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/opv2v/doma_codefilling/pretrain_m2.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/opv2v/doma_codefilling/pretrain_m3.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/opv2v/doma_codefilling/pretrain_m4.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/codebook.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/detector.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/finetune.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/infer.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/pretrain_m1.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/pretrain_m2.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/pretrain_m3.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/hypes_yaml/v2xset/doma_codefilling/pretrain_m4.yaml` | added | 独立实验/训练阶段配置，不覆盖官方 YAML |
| `opencood/modality_assign/opv2v_4modality.json` | added | 从 DOMA 原样复制的训练或 ordered assignment |
| `opencood/modality_assign/opv2v_4modality_in_order.json` | added | 从 DOMA 原样复制的训练或 ordered assignment |
| `opencood/modality_assign/v2xset_4modality.json` | added | 从 DOMA 原样复制的训练或 ordered assignment |
| `opencood/modality_assign/v2xset_4modality_in_order.json` | added | 从 DOMA 原样复制的训练或 ordered assignment |
| `opencood/models/codefilling_doma.py` | added | 独立四成员 CodeFilling 组装、训练冻结与真实 CPU 核心 |
| `opencood/models/fuse_modules/codefilling_max.py` | added | 从原生文件逐字提取的 regroup/MaxFusion |
| `opencood/models/heter_encoders.py` | modified | LSS nonpersistent device buffers；仅新模型 opt-in 使用 DOMA camera contract |
| `opencood/models/sub_modules/codefilling_camera.py` | added | 与 DOMA 源类 AST 一致的两个相机实现 |
| `opencood/models/sub_modules/codefilling_transport.py` | added | 原生码本/Filling 包装、实际 sampled codes 与多尺度字节账本 |
| `opencood/models/sub_modules/torch_transformation_utils.py` | modified | 移除未使用的 icecream import，使原生几何 warp 可做 CPU 契约测试 |
| `opencood/tools/audit_codefilling_doma.py` | added | 独立 audit_codefilling_doma 命令入口 |
| `opencood/tools/check_codefilling_doma.py` | added | 独立 check_codefilling_doma 命令入口 |
| `opencood/tools/inference_codefilling_doma.py` | added | 独立 inference_codefilling_doma 命令入口 |
| `opencood/tools/prepare_codefilling_doma.py` | added | 独立 prepare_codefilling_doma 命令入口 |
| `opencood/tools/train_codefilling_doma.py` | added | 独立 train_codefilling_doma 命令入口 |
| `opencood/utils/codefilling_protocol.py` | added | 来源/四模态/范围/服务器 receipt 契约 |
| `opencood/utils/codefilling_state.py` | added | 严格父权重所有权、完整 checkpoint 和 epoch 续训 |
| `opencood/utils/pcd_utils.py` | modified | 保留 RGB intensity，兼容显式 intensity 字段并拒绝无第四特征 |
| `tests/test_codefilling_doma.py` | added | Windows 静态与 CPU 实际算法/数据流/状态测试 |
