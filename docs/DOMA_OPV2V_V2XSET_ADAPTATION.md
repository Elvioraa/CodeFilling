# CodeFilling 对 DOMA 的双数据集适配报告

日期：2026-10-27（Asia/Shanghai）。仓库：`F:\zzuStudy\exp_gpt\Baselines\CodeFilling`。原始 HEAD：`57f83cc`；开始时 Git 工作区干净。直接使用现有仓库，未克隆、初始化、覆盖已有实验配置、提交、推送、部署或运行 GPU 训练。

## 1. 交付范围和验证边界

已实现独立四成员模型、双数据集配置、分阶段训练入口、固定 m1 ego 的 CAV1–4 推理入口、协议来源锁定、文件核验器和 Windows CPU 契约测试。原始 Codebook、Information Filling、官方模型、官方 YAML、损失与 AP 评估器保留。

本地通过的是静态协议与真实 CodeFilling 核心的 CPU 契约测试。没有真实数据、完整感知环境和训练权重，因此不提供虚构 AP，也不把本地通过解释为服务器 CUDA/AP 已对齐。两个数据集的端到端状态均为 `SERVER_VALIDATION_REQUIRED`。

## 2. 实际源码审查与原生训练依赖

| 原生文件 | 实际行为及适配处理 |
|---|---|
| `models/sub_modules/codebook.py` | `UMGMQuantizer` 使用多段、多层残差码本、hard Gumbel 样本和重建 MSE；所有算法代码保持原样。底部还实例化全局 `ChannelCompressor`；新入口在 import 完成后设 seed。 |
| `models/heter_model_shared_head_solver.py` | 四模态 encoder/backbone/aligner → 64 通道公共特征 → single heads → Codebook 重建 → ego 原特征旁路 → 多尺度特征 → mask → fusion → shared heads。新模型保留这一计算顺序。 |
| `models/comm_modules/where2comm.py` | 原生 Information Filling 对 confidence 做可选平滑、几何 warp、sum 或 max solver，再与置信度阈值相乘；ego mask 保持全 1。原算法未改，包括 sum 分支的严格累计阈值规则。 |
| `models/fuse_modules/fusion_in_one.py` | 新文件中的 `regroup`、`MaxFusion` 从本仓库原文件逐字提取，AST 一致；避免 CPU 测试导入无关的融合网络依赖。 |
| `tools/train.py` | 原生 detection/single loss 训练器不自动把 `codebook_loss` 加入总损失，末尾还会自动启动测试脚本。新入口明确阶段损失和验证集选权重，训练结束不运行测试集。 |
| `tools/get_feature.py`、`train_codebook_with_feature.py`、`codebook_trainer.py` | 存在固定特征目录、固定迭代/epoch、两模态参数以及不完整的续训路径，不能直接解释成可移植的四模态训练流水线。新入口在线提取本阶段感知特征，不依赖这些绝对路径或离线特征文件。 |
| `tools/inference_comm_bp_plus.py` | `--modal` 会把四成员重映射为两模态，并覆盖 range/communication；新评测入口保持四种编码器身份，不调用该参数覆盖逻辑。 |
| `models/heter_encoders.py` | PP、SECOND 与本地 DOMA 同类 AST 一致；LSS 使用相同几何与 voxel pooling。固定 CUDA 张量改为 nonpersistent buffers，随 `.to(device)` 移动，兼容原 state key 集。 |

原生 CodeFilling 的依赖关系是：先有可用检测特征 → 在这些特征上训练 Codebook → 用匹配的检测器与码本进行受通信约束的微调/评测。不能把随机码本配到已训练检测器上当正式 CodeFilling 结果，也不能仅用 detection loss 代替重建训练。

新增依赖链在每个数据集内独立：

```text
pretrain_m1 ─┐
pretrain_m2 ─┼─> detector ─> codebook ─> finetune ─> infer
pretrain_m3 ─┤
pretrain_m4 ─┘
```

- `pretrain_m1..m4`：独立对应成员，训练原生 detection、direction、single 以及有真实标签时的 depth loss；不启用通信筛选和量化。
- `detector`：按成员严格装载各自 encoder/backbone/aligner，公共 fusion/head 从 m1 预训练权重装载；Codebook 显式新初始化。随后联合训练公共特征/检测器。
- `codebook`：严格加载 detector 全状态；冻结感知、head、fusion 和 BN，仅训练原生 UMGM 重建 MSE。
- `finetune`：严格加载 codebook 全状态；冻结码本并保持 eval，启用原生量化与 Information Filling，使用检测损失微调其余网络。
- `infer`：只允许已验证并按验证损失选出的本适配 finetune checkpoint；严格装载全部模型状态。

每个阶段保存 `last.pth` 和验证集最优 `best.pth`；包括模型、optimizer、scheduler、epoch、best、Python/NumPy/Torch/CUDA RNG、DataLoader generator、配置 hash 和父权重 hash。续训仅支持 epoch 边界；CAV 重排按 seed+epoch 重建，避免构造器 RNG 影响续训顺序。CPU fixture 验证下一次参数更新可精确恢复；不宣称跨 CUDA/硬件位级一致。

训练需要各数据集自身的完整阶段权重，**不加载 DOMA 检测权重或 DOMA 对象/几何/共识模块**。m2 的 DOMA 同源 EfficientNet 实现调用 `EfficientNet.from_pretrained('efficientnet-b0')`，需要对应预训练缓存或服务器联网准备；m4 的 ResNet101 构造参数为 `pretrained=False`。没有下载或训练这些权重。

## 3. 协议依据及模态对齐

OPV2V 从 DOMA `V2_R2_TABLE4_EXT/history_snapshot.json` 中 Full `DOMA_V2_R2_2GPU_LR1P414_SEED303` 保存配置提取 m1、m2、m3、m4、final。V2XSet 从 DOMA `v2xset/MoreModality/DOMA/V2/` 对应五份本地 YAML 提取。原文、来源路径和 SHA256 保存在 `docs/codefilling_sources/`。V2XSet 本地配置不能冒充已完成的服务器运行记录。

| 成员 | 编码器 | OPV2V 输入 | V2XSet 输入 | 关键预处理 |
|---|---|---|---|---|
| m1 | PointPillar | 原始 PCD | 原始 PCD | voxel `[0.4,0.4,4]`，32 点/voxel |
| m2 | LSS EfficientNet-B0 | RGB+depth，4 cameras | RGB，4 cameras，无 depth 监督 | 384×512，LID `[2,50,48]` |
| m3 | SECOND | `OPV2V_Hetero/.../*_32.pcd` | **原始 `.pcd`，不生成 `_32.pcd`** | voxel `[0.1,0.1,0.1]`，5 点/voxel，XYZI 4 维 |
| m4 | LSS ResNet101 | RGB+depth，4 cameras | RGB，4 cameras，无 depth 监督 | 336×448，LID `[2,50,48]` |

相机类逐字提取 DOMA 对应实现，仅供新模型的 opt-in `camera_contract=doma` 分支使用。m4 的真实 DOMA 干路只使用 conv1/layer1/layer2，layer3 为 Identity；不能描述成完整分类 ResNet101 前向。原 CodeFilling m4 的两个未使用 Up 模块和 in-place ReLU 没有混入协议实现。m2 类的局部变量命名也与来源一致。

所有四成员输出在 0.8 m 物理栅格上用中心裁剪/补零统一范围，不进行坐标空间拉伸。PP 的 `.4×2`、SECOND 的 `.1×8` 和相机的 `.4×2` 在 XY 上对齐。没有插入额外学习式替代融合器。

| 设置 | OPV2V 来源 | V2XSet 来源 |
|---|---|---|
| m1 预训练范围 | `[-102.4,-51.2,-3,102.4,51.2,1]` | 相同 |
| m2/m4 预训练范围 | `[-51.2,-51.2,-3,51.2,51.2,1]` | 相同 |
| m3 预训练范围 | `[-96,-48,-3,96,48,1]` | 相同 |
| final YAML 范围（作为联合阶段本地起点） | `[-102.4,-102.4,-3,102.4,102.4,1]` | `[-102.4,-51.2,-3,102.4,51.2,1]` |
| YAML 通信范围 | 70 m | 70 m |
| NMS / score | 0.15 / 0.2 | 0.15 / 0.2 |
| final GT max_num | 100 | 150 |
| GT max_cav | 5 | 5 |

最终推理范围和通信范围必须从实际 DOMA 服务器调用记录核验。虽然本地 ordered entry 写有额外默认值，这里没有把它们当作已核验服务器参数。`docs/server_settings.example.yaml` 故意保留空值与 `SERVER_VALIDATION_REQUIRED`；评测需要独立填写 `status: VERIFIED`、实际证据、数据集、六维 `eval_range`、`comm_range` 和 `test_dir`。针对性复核已将 `doma_formal_evaluation` 设为强制入口：必须记录正式 DOMA 命令与证据，其三个实际参数须与 CodeFilling receipt 逐项相同；V2XSet 还必须有正式排除场景清单和证据。缺失/不一致时，在读取权重和构造模型前报 `SERVER_VALIDATION_REQUIRED`。这只是显式证据入口，工具不能自动保证人工证据真实。

16 个实验 YAML 为独立适配模板，batch=1、epoch=30；优化器与 scheduler 初值来自相应 DOMA 来源。码本 `seg_num=2, dict_size=64, r=3` 与原生构造默认一致，solver 为 sum/1.0，confidence threshold=0。**batch=1、epoch=30 是暂定值，保持可编辑；未经用户审核，不得直接作为正式训练预算执行。**它们不是服务器最优设置或已复现的论文预算。预算、阈值、显存和正式选参均为 `SERVER_VALIDATION_REQUIRED`，不能用测试集 AP 选参。本轮未改这 16 份 YAML 的预算和训练链。

## 4. 数据加载和 CAV1–4 评测

Assignment 从 DOMA 原 JSON 原样复制，train 与 ordered 两套分别保存。ordered 场景数为 OPV2V 16、V2XSet 19；它们不是数据集帧数。测试中按 DOMA 的排序规则把负 ID 路侧节点移至末尾，并校验 m1→m2→m3→m4 循环身份。成员缺失、映射不符或目录集合不一致时报错，不补造 CAV。

新增 Dataset 使用 DOMA 原始 ordered inference `__getitem__`：先汇总通信范围内最多 5 个成员的 GT，再对输入做 `use_cav` 截断。m1 ego 固定；CAV1、CAV1+CAV2、CAV1+CAV2+CAV3、CAV1+CAV2+CAV3+CAV4 使用同一测试帧与 GT。针对性复核补齐了实际场景/时间戳身份：跨组逐帧比较真实 `scene + timestamp + sample_idx + all_cav_ids + GT hash`，不再仅依赖索引与 GT hash。保持 DOMA `visualize=True` 的 GT 选择行为，四槽位都用 LiDAR 标签。原生 CAV 不足 n 的帧保留，realized 必须等于 `min(requested, 可用 CAV 数)`；验证真实 CAV ID 前缀与 m1→m2→m3→m4，并将每条通信 sender 绑定到实际 CAV ID/模态。范围过滤造成非连续模态前缀时拒绝整次结果，不偷偷重新命名。

评测复用 CodeFilling 原 `eval_utils.caluclate_tp_fp` 和 `calculate_ap`，输出 `AP@0.3`、`AP@0.5`、`AP@0.7` 至 CSV/JSON，并保存各阈值原始 TP/FP/score/GT、每帧成员列表、GT hash 和通信账本。空数据/空 GT/丢帧/跨组 GT 不一致均拒绝汇总。当前没有真实 AP 文件。

基础加载器补齐了 `modality_name` 传播、Windows basename、缺省 augmentation，以及 RGB-only HDF5 不访问 depth 的行为。PCD 保留原 RGB intensity 分支；无 colors 时读取真实 `intensity` 字段，不填假特征，不猜归一化。该新增字段读取的服务器数值尺度需要与实际 DOMA 数据流对照。OPV2V-H 文件不存在就报错，V2XSet 始终使用原 PCD。

V2XSet 缺相机场景 `2021_09_09_13_20_58` 按原类排除；既有工厂在所有阶段保持相同排除策略。已复核本地 DOMA `v2xset_basedataset.py`：`load_camera_file=True` 或 `v2xset_camera_compatible=True` 时排除该场景，final YAML 含 camera；`docs/V2XSET_DOMA_ADAPTATION.md` 的本地说明也一致。但这只能证明本地源码行为。已搜索本地 DOMA 保存的 V2XSet 配置、说明及 logs/temp 中的记录，未取得正式 V2XSet 评估命令和服务器测试样本清单；文件清单中出现 v2xset 源码路径不能作为正式评测证据。因此，**DOMA 正式评估是否排除该场景仍是运行前阻塞项 `SERVER_VALIDATION_REQUIRED`**；本轮不增删数据、不改变 Dataset 场景策略。服务器需提供正式命令、版本/配置及实际 scene/frame 清单，证明排除政策一致，才能通过 receipt。

真实数据核验工具遍历所有 split/scene/frame/CAV，检查 assignment、原 PCD、OPV2V `_32.pcd`、4 cameras、depth/visibility、HDF5 key、pose/标定和 PCD header。默认全量；`--frames-per-scene > 0` 仅报告采样通过。它不替代真实 decoder/标定数值/稀疏卷积前向验证。

## 5. 通信量定义与可核验性

原生 `comm_rates` 只表示 mask 占比，不能直接当通信字节。新账本排除每个 batch 的 local ego；每条 peer→ego 链路分别取整并列出：

1. 最细尺度实际 sampled code indices：`selected_nonzero_cells × segments × levels × ceil(log2(K))` bits，逐消息 `ceil(bits/8)`。
2. 一个 H×W 位图，以及几何 warp 后非 0/1 mask 的 FP32 权重。按 nonzero support 计数，避免把分数 mask 的 sum 当成发送 cell 数。
3. 原生在 mask 前生成粗尺度，细尺度稀疏码字不能独自重建粗特征；因此另计各 pooled mask 非零位置的 **原生粗尺度 FP32 特征向量**。没有把粗特征当作免费的通信。
4. 控制流另计每 peer 一张 dense FP32 confidence 与六个 FP32 pose 数值。位图/分数 mask 对应选择反馈，已计入总字节。共享码本预安装；一次性 checkpoint 分发另行处理。

原生 quantizer 返回 argmax codes，但 decoder 实际使用 hard Gumbel 样本；新代码从真实 decoder 输入捕获码字，不改变其随机算法。CPU 测试证明这些码字经过位打包/解包及原生 decoder 后，与实际 peer 重建一致；原生重建损失也一致。没有修补未被主路径使用的 `_ix` decode API，也没有声称原生 entropy coder 已可用。

### 实际执行路径复核

最终 `infer` 的单体 confidence 从**量化前**的四成员特征生成。包括 ego 在内的所有输入特征先执行原生 UMGM 全图量化、逐层 Gumbel 采样和重建；然后用原特征替回每个 batch 的 ego。因此，ego 确实参与码本计算和随机数消耗，但重建结果不参与融合、也不记为发送消息。所有 peer 均先重建，之后原生 Communication 根据原始 confidence 生成 mask；未选位置置零。原生多尺度 backbone 在 mask 前计算，三尺度分别乘同一 mask 的逐级 max-pool 结果，再几何 warp 和 MaxFusion。

前向钩子测试覆盖默认码本参数下的 dense、全抑制和非零平移场景：直接记录 quantizer 输入、confidence、实际 reconstruction、每个 fusion 输入和输出；账本 selected cells、真实 sampled index 打包字节、粗尺度实际通道/元素字节与这些前向张量逐项一致。sender 列表必须与真实输入前缀一致，ego 不得进入发送账本。保持了逻辑消息记账；模型仍是集中式 tensor 模拟，没有真实网络 send/receive。

默认 `confidence threshold=0` 没有关闭 Information Filling。对通常为正的 sigmoid confidence，该阈值本身不再稀疏化；但 `sum solver / thre=1` 仍先按 peer confidence 降序排序，加上 ego 后计算累计值，保留的是**累计值严格小于 1 的位置**。跨过/刚好达到 1 的 peer 自身也不选：`[ego=.5, peer=.5]` 不选 peer；`[.1,.8,.6,.3]` 只选 .8；`[.1,.6,.2,.05]` 全选三个 peer。极小 logits 可能使浮点 sigmoid 下溢为 0，仍会被 `>0` 排除。原算法与默认参数均未改动。

一个容易误判的原生行为是：peer mask 全零不等于从 fusion 中删除该 peer。其零张量仍参与 MaxFusion，在覆盖区会把 ego 的负特征与 0 取 max。因此，“全部 peer 发送 cell 为零”的输出不保证等于 CAV1；控制消息也仍可能非零。测试按原生 warp 后 `max(ego,0)` 核验这个行为，没有为追求 CAV1 一致性而修改算法。

### 随机性复核

新 Python 进程测试确认，首次 import `codebook.py` 时实例化全局 `ChannelCompressor` 确实消耗 Torch RNG。适配入口在核心模块 import 后设 seed，测试证明先单独 import 码本或经核心 import 后，再设 seed=303，核心模型 state hash 相同。完整传感器导入/构建还需服务器环境验证。

原生 `_sample` 和 `gumbelSoftmax` 没有 `.training`/`.eval` 的确定性分支，`eval()` 仍调用 `torch.rand_like` 和 hard Gumbel。不重置 RNG 连续推理，sampled codes 会变化；peer 被选中时，检测输出也可能变化。CAV1 虽然仍消耗采样 RNG，但 ego bypass 后输出在固定输入/权重下不因码本采样改变。相同输入/模型、重置相同 RNG 的 CPU 推理结果和 sampled codes 可复现。

修复了采样起点受初始化影响的隐患：每个 CAV 组在模型与 Dataset 构造完成、DataLoader iterator 创建之后才重置 seed=303；DataLoader 使用独立 generator。**不逐帧重置 seed，不改为 argmax，不修改原生采样机制。**相同完整运行条件下可重复；CAV 数不同会改变随机数消耗量，不承诺跨 CAV 组的某个 peer 使用相同随机噪声，也不承诺 CUDA 跨硬件/版本位级一致。

账本标记 `logical_packed_protocol_v1`，提供每消息最细/粗尺度/control 字节和总和。它是可复算的逻辑通信协议计数，**不是已实测网络流量、吞吐或时延**；传输层、包头、重传、码本首次分发未包括。几何/标定假定已知；若实际服务器另传 calibration 等信息，需另计。账本同时列出 dense feature 基准和原生 comm_rates，不能用后者替代总字节。

## 6. 本地验证和命令

在本仓库根目录运行；测试临时目录放仓库内，避免本机系统 Temp ACL 问题。这里只执行了静态、CPU 合成特征及单元 fixture，没有执行完整感知模型训练。

```powershell
New-Item -ItemType Directory -Path .test_artifacts -Force | Out-Null
python -B -m pytest -q tests/test_codefilling_doma.py -p no:cacheprovider --basetemp .test_artifacts/review_01
python -B -m opencood.tools.check_codefilling_doma --output docs/codefilling_local_checks.json
python -B -m opencood.tools.check_codefilling_doma --check-data --output docs/codefilling_data_checks.json
```

静态核验通过两数据集全部 16 份实验 YAML，源 sensor/model/preprocess/postprocess/range/input 配置逐项比较。针对性复核后 CPU 测试 **38 passed**，覆盖：真实 UMGM 前后向/损失、实际 sampled codes 解码与位打包、原生 sum/max Filling、ego 零通信、fractional masks、粗尺度字节、物理栅格、真实三尺度 backbone/fusion/head、冻结和 BN、所有配置核心构建、父阶段所有权、缺失权重拒绝、完整状态续训、原 ordered GT-before-cutoff、基础 modality/HDF5 和 PCD 分支、原生 AP 全局排序；新增默认 sum/threshold=0 的边界行为、实际三尺度消息钩子、Gumbel 重复推理、fresh-import RNG、初始化后采样 seed、真实 scene/timestamp/GT/prefix/sender 身份、规范输出目录及预算可编辑测试。18 个新增/修改 Python 文件编译通过，10 个原生关键文件与初始 Git 版本内容一致（新增核验实际 hard-Gumbel 所在 `codebook_utils.py`；按 Windows/Git 换行约定比较），`git diff --check` 通过。完整测试输出见 `codefilling_test_results.txt`。

本机 Python 3.12.4、Torch 2.7.0+cpu。缺 spconv、torch_scatter、timm、mmcv、icecream、tensorboardX 等完整感知依赖；数据目录不存在。`--check-data` 正确报告 `SERVER_VALIDATION_REQUIRED` 并非零退出。原生 CUDA NMS/IoU 与相机、SECOND、PCD/HDF5 实际 decoder 必须服务器验证；不以 fixture 替代。

以下是待审核后在有完整环境的机器上使用的命令形式，**本轮未执行**。每个数据集使用独立输出目录；`$ds` 分别取 `opv2v` 与 `v2xset`：

```powershell
$ds = 'opv2v'
$cfg = "opencood/hypes_yaml/$ds/doma_codefilling"
$run = if ($ds -eq 'opv2v') { 'opencood/logs/CodeFilling_OPV2V_SEED303' } else { 'opencood/logs/CodeFilling_V2XSET_SEED303' }
foreach ($m in @('m1','m2','m3','m4')) {
  python -B -m opencood.tools.train_codefilling_doma --config "$cfg/pretrain_$m.yaml" --output "$run/pretrain_$m"
}
python -B -m opencood.tools.train_codefilling_doma --config "$cfg/detector.yaml" --output "$run/detector" --parent "pretrain_m1=$run/pretrain_m1/best.pth" --parent "pretrain_m2=$run/pretrain_m2/best.pth" --parent "pretrain_m3=$run/pretrain_m3/best.pth" --parent "pretrain_m4=$run/pretrain_m4/best.pth"
python -B -m opencood.tools.train_codefilling_doma --config "$cfg/codebook.yaml" --output "$run/codebook" --parent "detector=$run/detector/best.pth"
python -B -m opencood.tools.train_codefilling_doma --config "$cfg/finetune.yaml" --output "$run/finetune" --parent "codebook=$run/codebook/best.pth"
python -B -m opencood.tools.inference_codefilling_doma --config "$cfg/infer.yaml" --checkpoint "$run/finetune/best.pth" --server-settings "server_settings_$ds.yaml" --output "$run/infer"
```

一阶段一进程；拒绝 DDP。`--check-only` 可在 CPU 上核验训练配置/父文件，不构造感知网络。新阶段输出目录已存在时拒绝覆盖；续训使用相同配置加 `--resume path/to/last.pth`，不是把原生 checkpoint 随意重命名为 `best.pth`。

正式输出根固定为 `opencood/logs/CodeFilling_OPV2V_SEED303/` 和 `opencood/logs/CodeFilling_V2XSET_SEED303/`。每个现有阶段单独使用 `pretrain_m1..m4/`、`detector/`、`codebook/`、`finetune/`、`infer/` 子目录。入口的 `--output` 可省略；显式传入时必须与该阶段规范路径一致，禁止自动追加时间戳。训练阶段各有 `config.yaml`、`epochs.jsonl`、`last.pth`、`best.pth`；推理阶段保存自己的有效 `config.yaml`、`run.json`（权重 hash/receipt/seed 规则）、四组 frame/raw AP 日志及 results。此次未创建正式实验目录、未训练。以上命令须先审核暂定预算与服务器验收记录后再执行。

## 7. 剩余问题

- 服务器实际数据身份、split/frame 数、所有原始/32线文件和标定、depth/visibility/HDF5 全量可用性。
- 实际 DOMA 推理 CLI 的 eval_range、comm_range、GT/NMS/AP 设置与新 receipt 对照；本地 YAML 已保存来源，不能替代正式运行证据。
- V2XSet 正式评估对 `2021_09_09_13_20_58` 的排除政策与实际测试 scene/frame 清单，现仅有本地源码/说明依据，是运行前阻塞项。
- 完整四传感器 CUDA 前后向、真实 loss、显存、正式训练预算/通信阈值选择、验证集选模型、最终 AP 与速度测量。
- Codebook 全图多层量化可能占用较大显存；没有服务器实测，不承诺 batch=1 即可运行所有范围。
- 原始 CodeFilling 论文的发布预算/预训练权重清单在当前仓库中不完整；本适配提供显式可训练依赖链，不能声称已复现论文数值。
- 没有网络传输服务；逻辑账本与实测链路结果需分开报告。

全部源码改动与新增文件清单由 Git 实际工作区生成，见 `CODEFILLING_CHANGED_FILES.md` 和 `codefilling_changed_files.json`。关键算法的原始 Git blob 与当前 hash 对照见 `codefilling_native_audit.json`。等待用户审核，不执行 push、部署或 GPU 训练。

## 8. 本轮针对性复核的实际修改

本轮在上轮 60 个变更文件的基础上只修改以下 12 个文件，未新增文件。保持现有训练链、16 份实验 YAML、Dataset 筛选策略、CodeFilling 核心组装/transport、原生 Codebook/Information Filling/MaxFusion/采样机制不变；已用复核开始时的文件 hash 再确认这些边界。

| 文件 | 本轮实际修复或补充 |
|---|---|
| `opencood/utils/codefilling_protocol.py` | 固定阶段输出目录；实际 scene/timestamp/GT/prefix/sender 校验；正式 DOMA 命令/范围/通信范围/测试目录与 V2XSet 排除证据的运行前门槛 |
| `opencood/tools/inference_codefilling_doma.py` | 初始化后启动 seed303 原生采样流，独立 DataLoader generator；记录实际帧身份及物理消息 sender；规范输出路径及推理独立 config/run 日志 |
| `opencood/tools/train_codefilling_doma.py` | 仅调整规范输出路径，`--check-only` 显示阶段目录；不改变训练阶段、损失、冻结策略或预算 |
| `opencood/tools/check_codefilling_doma.py` | 静态结果明确列出正式命令一致性、V2XSet 正式样本排除证据的运行前阻塞，以及暂定预算待审核 |
| `opencood/tools/audit_codefilling_doma.py` | 将实际 hard Gumbel 实现 `codebook_utils.py` 纳入原生算法零差异审计 |
| `docs/server_settings.example.yaml` | 增加强制正式 DOMA 命令、匹配参数、excluded scenes 和场景政策证据占位，不填写服务器假参数 |
| `tests/test_codefilling_doma.py` | 增加执行路径、消息、默认 sum 规则、零张量 max 行为、重复推理、导入 RNG、样本身份、规范目录和预算可编辑测试 |
| `docs/DOMA_OPV2V_V2XSET_ADAPTATION.md` | 更新本报告，说明实际行为、修复项、正式目录、暂定预算和验收阻塞 |
| `docs/codefilling_local_checks.json` | 更新两数据集静态检查与运行前阻塞记录 |
| `docs/codefilling_data_checks.json` | 更新数据核验记录；本地数据缺失，退出码 2，未伪报通过 |
| `docs/codefilling_native_audit.json` | 更新 10 个关键原生文件零差异及 18 个 Python 文件编译结果 |
| `docs/codefilling_test_results.txt` | 完整 CPU 回归：38 passed |

尚需服务器验收：正式 V2XSet scene/frame 清单及指定场景排除证据；两数据集正式 DOMA 命令与实际范围/通信/测试路径一致性；完整传感器 CUDA 前后向、采样重复性和 AP；真实 PCD/depth/visibility/HDF5 核验。正式预算需用户审核。上述事项未验证前，本地 `PASS_STATIC`/CPU 通过不构成正式实验放行。

本轮仅执行只读证据审查、静态检查及 CPU 契约测试，未提交、推送、部署或启动实验训练；没有创建正式实验输出目录或时间戳目录。

## 9. GitHub 发布准备（2026-10-31）

用户授权将现有仓库与适配上传至 `Elvioraa/CodeFilling`。再次运行 CPU 测试，38 passed；双数据集静态检查通过。发布新增 `.gitattributes`，使协议参考副本、assignment 和提取实现保留原始换行字节，防止 Git 自动换行转换破坏已锁定的 SHA256；提取实现的源格式保留，不清理其继承的行尾空白。已验证暂存对象与锁定参考文件逐字节相同。发布范围为此前清单中的 60 个文件，加此 Git 属性文件共 61 个；不包含数据、checkpoint 或本地实验输出。前文“未提交、未推送”是前两轮复核结束时的记录；本轮按用户的新授权进行提交和上传。服务器待验证项保持不变。
