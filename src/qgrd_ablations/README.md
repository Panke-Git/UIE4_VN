# QGRD：基于 canonical V16 的四套独立消融

这四套实验以原 V16 的三集合协议为基础，不使用 V19-SDQR、V18-NLQC 或 V16-19 的两集合协议。所有实验从头训练，不加载已有 V13/V16/V19 checkpoint 作为初始化。

## A2 新实验：FP32 评价修复

在旧 A2 的 Epoch 124 验证中，已定位到 decoder3 第一层 Conv2d 的 FP16 输出溢出：
输入和权重有限，输出出现 +Inf；同一权重和 batch 的 FP32 forward 有限。
本次按用户选择，仅修复 A2（LSUI 与 UIEB）：训练仍为原 AMP，
验证和测试的模型 forward/loss 固定 float32 并关闭 autocast。
不改模型、UICF、Query assignment、loss、optimizer、学习率、batch、增强或清单。
FP32 评价不额外改变 PyTorch 的 TF32 backend 设置。

两份 shared_query.yaml 新增 evaluation.precision: fp32，实验名带 FP32Eval。
启动方式不变，自动创建新的 run 目录，不覆盖旧 run：

```bash
python -m src.qgrd_ablations.shared_query.train --config configs/ablations/qgrd/lsui/shared_query.yaml --seed 3520 --gpu 0
python -m src.qgrd_ablations.shared_query.train --config configs/ablations/qgrd/uieb/shared_query.yaml --seed 3520 --gpu 0
```

此次从 Epoch 1 开始，**不要添加 --resume 指向旧 Epoch 123 run**。
新 run 的中断恢复仍使用原 --resume <NEW_RUN>/checkpoint/last.pt。
旧 run 的配置/checkpoint/history 不修改；新代码明确拒绝缺少 FP32 声明的旧 A2 配置，
且保留源码 SHA256/精度协议检查，不迁移或重新标记旧 checkpoint。
如需将来运行旧实验，应保留并使用它记录的旧源码 checkout。

部署到服务器至少同步整个 src/qgrd_ablations/shared_query/（不要上传 __pycache__），
以及 configs/ablations/qgrd/lsui/shared_query.yaml 和
configs/ablations/qgrd/uieb/shared_query.yaml；汇总时也同步 tools/qgrd_ablation_report.py。
只上传之前的诊断脚本不会修复正式训练入口。

run_info、checkpoint、best JSON、validation_summary 和 test_summary
均记录 evaluation_precision / evaluation_forward_protocol。
evaluation.py 仅供 A2 自己调用，其他变体没有 import 它。
PSNR/SSIM/DeltaE00 算法、float-before-PNG、尺寸、crop 和 checkpoint selector 均不变。

**本轮未改变 V16、A1、A3、A4 的评价执行精度。**
qgrd_ablation_report.py 会拒绝把新 A2 FP32 结果和旧 native AMP 评价协议直接混比；
不能仅改其他旧 run 的 YAML/summary 标签来绕过审计。后续统一论文精度属于另一项显式修改。
本轮不自动重训其他组或对真实实验进行测试。

本次数值修复验证：专项 9 passed；QGRD 73 passed；全仓库 414 passed，
0 failed/skipped（全仓库旧可视化依赖仍使用项目外临时目录）。
覆盖 FP16 Conv 溢出回归、嵌套 autocast 关闭、FP32 指标与参考实现一致、
新 run train/test/resume 和精度 metadata、旧 run 拒绝与不覆盖、混合精度汇总拒绝。
训练循环 AST 与原 V16 完全一致，模型/shared/其他消融源码不变。
本地无 CUDA，未自动启动云端真实 200 epoch 训练。

## 数学定义与解释边界

完整 V16：
`T0 → T4 → T3 → T2 → T1`，每级 decoder 使用
`D̂s = Rs(Gs(Ds, Ts))`。UICF 始终放在 backbone 之前。

| 实验 | 固定结构 | 解释边界 |
|---|---|---|
| A1 / no_qgrd | UICF → 原始 Plain U-Net；无 Query、Guidance、Local Refine | 完整 QGRD 模块的总体作用，不是单独 attention 的作用；有效结构等价 V13，但必须独立训练 |
| A2 / shared_query | 保留完整 query 链和四级 G/R，所有 G 均接收最终 T1 | 只改变 stage-to-query assignment，容量不变 |
| A3 / single_stage | 保留完整 query 链、四级 R；只保留并调用 D1 的 G1 | 对照 multi-stage guidance；删除三级 attention 后容量下降，不能排除容量因素 |
| A4 / no_local_refine | 保留完整 query 链和匹配尺度的 G；无四级 R | Local Refine 的整体贡献，容量下降；不能直接归因于某种特定空间机制 |

| Component | V16 | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|
| UICF + Plain U-Net | ✓ | ✓ | ✓ | ✓ | ✓ |
| Progressive queries | ✓ | × | ✓ | ✓ | ✓ |
| Guide4/3/2 | ✓ | × | ✓ | × | ✓ |
| Guide1 | ✓ | × | ✓ | ✓ | ✓ |
| Refine4/3/2/1 | ✓ | × | ✓ | ✓ | × |
| Guidance token assignment | T4/T3/T2/T1 | — | T1/T1/T1/T1 | 仅 G1(T1) | T4/T3/T2/T1 |

A4 不删除 SpatialTokenGuidance 内部的 `spatial_feature + guided` residual。
A3 不删除前三层 local refine。不存在动态 mode 切换、forward hook 或 monkey-patch 实现。

不得根据观察结果修改消融定义、筛选有利图像或隐瞒消融优于 Full V16 的结果。

## 实现与初始化公平性

四套独立源码：

- `src/qgrd_ablations/no_qgrd/`
- `src/qgrd_ablations/shared_query/`
- `src/qgrd_ablations/single_stage/`
- `src/qgrd_ablations/no_local_refine/`

每套都有自己的 dataset / engine / experiment / losses / metrics / utils /
train / test / provenance，以及 models/network.py 和 models/backbone.py。
四套之间无相互 import；没有公共 ablation engine。原 shared、V4 等来源仅作为只读依赖。

每个 backbone 都先完整调用原 PlainUNetColorQuery 初始化，再删除需要消融的注册模块；
之后构造完全相同的 canonical UICF。删除模块不消耗 RNG，最终模型不保留死参数。
同 seed 下所有仍保留的参数和 buffer，包括 UICF，与 V16 tensor-exact。

标准配置（base_channels=64、token_dim=128、M=8）的参数量：

| 模型 | Total | Trainable | 相对 V16 |
|---|---:|---:|---:|
| Full V16 | 38,878,217 | 38,878,217 | 0 |
| A1 | 31,175,497 | 31,175,497 | −7,702,720 |
| A2 | 38,878,217 | 38,878,217 | 0 |
| A3 | 38,449,417 | 38,449,417 | −428,800 |
| A4 | 32,607,497 | 32,607,497 | −6,270,720 |

各包 `frozen_dependencies.json` 记录只读依赖源文件 SHA256；
开始训练、恢复训练和显式测试前均核对。run_info 和 checkpoint 另外记录本实验包源码、
版本/model type、数据协议及清单身份。部署时上传本次完整同版本代码和原清单，不能只上传新配置而沿用不同版本 shared。

## 八份配置与严格三集合协议

每个目录都含 no_qgrd.yaml / shared_query.yaml / single_stage.yaml / no_local_refine.yaml：

| 配置目录 | Dataset | Train / Validation / Test | 默认 root |
|---|---|---|---|
| configs/ablations/qgrd/lsui/ | LSUI19 | 3466 / 385 / 428 | /root/autodl-tmp/pro/publicdata/LSUI19_dup_train |
| configs/ablations/qgrd/uieb/ | UIEB | 720 / 80 / 90 | /root/autodl-tmp/pro/publicdata/UIEB19 |

LSUI 从 configs/config_v16.yaml 复制；UIEB 从 configs/uieb/config_v16_uieb.yaml 复制。
初始版本仅改变 experiment.version/name/output_root/model.type。
当前 A2 按上方数值修复额外声明 evaluation.precision=fp32；训练及其余协议仍逐项一致。
原始 split/lsui19/*.tsv 与 split/uieb/*.tsv 的 SHA256 固定；
不重划分，不使用 split/v16_19，不自动寻找替代数据。

训练保持 seed=3520、200 epochs、patch=256、batch=16、workers=4、原三种增强、
Charbonnier ε=0.001、AdamW lr=0.0002 / wd=0 / betas=(0.9,0.999)、
无 scheduler、CUDA AMP、无 gradient clipping、每 epoch 验证、每 10 epoch 周期保存。
CPU 下 AMP 按原逻辑关闭。

## 环境与验证

以下命令在云服务器的 UIE4_VN 根目录、原 PyTorch 环境中运行：

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
bash scripts/qgrd_ablations/verify_all.sh
```

复用项目 requirements.txt；单元测试还需 pytest。旧可视化回归测试可能需要
Matplotlib，缺失时应安装到你的测试环境，不能通过 skip/xfail 绕过。
脚本的 Python 默认是当前 `python`，可用 `PYTHON_BIN=/absolute/path/python` 指定。

verify_all 运行新增测试和要求的 V16/UICF/V19 regression tests，包含旧已跟踪文件 SHA256 保护检查。
`protection_snapshot.json` 是本次开始时的 Git 状态和全部旧已跟踪文件哈希。
运行 smoke 使用临时合成图像/清单，不覆盖真实 experiments/split。

## 最简单的一键训练

先检查将运行的命令（不构建模型、不训练）：

```bash
bash scripts/qgrd_ablations/run_lsui_all.sh --dry-run
bash scripts/qgrd_ablations/run_uieb_all.sh --dry-run
```

按 A1 → A2 → A3 → A4 顺序执行，上一组退出码非零立即停止；不后台并发：

```bash
bash scripts/qgrd_ablations/run_lsui_all.sh --gpu 0
bash scripts/qgrd_ablations/run_uieb_all.sh --gpu 0
```

使用默认数据路径不需要 --data-root。路径不同时，统一覆盖四组：

```bash
bash scripts/qgrd_ablations/run_lsui_all.sh --gpu 0 --data-root /absolute/path/to/LSUI19_dup_train
bash scripts/qgrd_ablations/run_uieb_all.sh --gpu 0 --data-root /absolute/path/to/UIEB19
```

脚本重新启动会创建新的独立 run；不会自动寻找旧 run 或自行恢复。
有中断时先用下述明确 checkpoint 恢复对应实验，再独立启动尚未训练的组，避免重复训练。

## 四个独立 LSUI 训练入口

```bash
python -m src.qgrd_ablations.no_qgrd.train --config configs/ablations/qgrd/lsui/no_qgrd.yaml --seed 3520 --gpu 0
python -m src.qgrd_ablations.shared_query.train --config configs/ablations/qgrd/lsui/shared_query.yaml --seed 3520 --gpu 0
python -m src.qgrd_ablations.single_stage.train --config configs/ablations/qgrd/lsui/single_stage.yaml --seed 3520 --gpu 0
python -m src.qgrd_ablations.no_local_refine.train --config configs/ablations/qgrd/lsui/no_local_refine.yaml --seed 3520 --gpu 0
```

UIEB 对应把四个 config 的 /lsui/ 换成 /uieb/；模块入口不变。
也可直接使用 run_uieb_all.sh，无需修改配置。

## 独立恢复

```bash
python -m src.qgrd_ablations.no_qgrd.train --resume "<A1_RUN_DIR>/checkpoint/last.pt" --gpu 0
python -m src.qgrd_ablations.shared_query.train --resume "<A2_RUN_DIR>/checkpoint/last.pt" --gpu 0
python -m src.qgrd_ablations.single_stage.train --resume "<A3_RUN_DIR>/checkpoint/last.pt" --gpu 0
python -m src.qgrd_ablations.no_local_refine.train --resume "<A4_RUN_DIR>/checkpoint/last.pt" --gpu 0
```

恢复读取冻结 config_resolved.yaml 和 split_snapshot，恢复 model(strict=True)、optimizer、
AMP scaler、Python/NumPy/Torch/CUDA RNG、DataLoader generator。
核对 checkpoint.version/model_type/resolved_config/variant/dataset protocol/source hashes。
A2→A3 或 LSUI→UIEB 的训练恢复均拒绝。除显式 --data-root 的路径迁移外，不允许改保存的配置。
数据迁移时须使用相同图像和配对。

用最近完整保存的 last.pt，checkpoint epoch 必须等于已有历史最后一轮；
拒绝将更老 periodic/best checkpoint 直接追加到更新历史，以免重复/混合 epoch。
已完成 200 epochs 的 run 不再 resume；直接显式 test。
训练完成不会自动测试。

## 四个独立测试入口

始终明确 run-dir，论文对照统一使用 best_psnr：

```bash
python -m src.qgrd_ablations.no_qgrd.test --run-dir "<A1_RUN_DIR>" --checkpoint best_psnr --gpu 0
python -m src.qgrd_ablations.shared_query.test --run-dir "<A2_RUN_DIR>" --checkpoint best_psnr --gpu 0
python -m src.qgrd_ablations.single_stage.test --run-dir "<A3_RUN_DIR>" --checkpoint best_psnr --gpu 0
python -m src.qgrd_ablations.no_local_refine.test --run-dir "<A4_RUN_DIR>" --checkpoint best_psnr --gpu 0
```

必要时各命令追加 --data-root /absolute/path/to/dataset。
诊断可沿用 V16 的 best_ssim / best_loss / last / 指定本 run 内 checkpoint 路径，
但汇总工具只接受明确 best_psnr 结果。

一键测试要求四个明确的路径，不自动匹配最新 run：

```bash
bash scripts/qgrd_ablations/test_lsui_all.sh "<A1_RUN_DIR>" "<A2_RUN_DIR>" "<A3_RUN_DIR>" "<A4_RUN_DIR>" --gpu 0
bash scripts/qgrd_ablations/test_uieb_all.sh "<A1_UIEB_RUN>" "<A2_UIEB_RUN>" "<A3_UIEB_RUN>" "<A4_UIEB_RUN>" --gpu 0
```

支持 --dry-run 和 --data-root。脚本核对数据集，各独立 test 入口核对自己的版本/模型/checkpoint。
测试只读该 run 的冻结 split_snapshot，不读取当前项目清单作为替代。

指标算法完全沿用 V16 native float protocol（A2 模型 forward 精度见上方修复说明）：

- PSNR ↑ / SSIM ↑ / ΔE00 ↓，预测转 float32 并 clamp 到 [0,1]，在 PNG 保存前计算。
- evaluation resize=true / size=256；crop_border=0；SSIM window=11 / sigma=1.5。
- 每张图等权平均；E00 用原 shared CIEDE2000 路径，D65/2°、kL=kC=kH=1、逐像素再逐图聚合，autocast 外运行。
- E00 耗时不计入模型推理耗时；不新增 best_e00，不加入 loss。
- 本轮不接入 PNG8/LPIPS/U45，不修改 paper evaluation whitelist。

重跑 native test 沿用 V16 的同名文件覆盖行为：更新 result/test_metrics.csv、
test_summary.json、test_visualization_samples.json、增强图、样例图和 grid，追加 test.log。
它不是事务式图像导出；运行途中失败可能留下部分本次图片。重跑相同 run/selector 即可，
需保留此前结果时先备份 result 和 log/test.log；不要混用不同 selector 的输出做论文汇总。

## 输出目录

```text
experiments/qgrd_ablations/<variant>/<lsui|uieb>/<version_name_seed_timestamp>/
  config_source.yaml / config_resolved.yaml / config.json
  run_info.json / status.json
  split_snapshot/train.tsv / validation.tsv / test.tsv
  checkpoint/last.pt / epoch_*.pt
  best/best_psnr.pt / best_ssim.pt / best_loss.pt (+ JSON)
  log/train.log / val.log / metrics_history.csv / metrics_history.json / test.log
  result/validation_metrics.csv / validation_summary.json
         test_metrics.csv / test_summary.json
         test_all_enhanced/ / test_samples/
         test_visualization_samples.json / test_grid_*.png
```

四个 variant × 两个 dataset 的 output_root 不重叠；时间戳包含微秒以避免启动碰撞。
训练和测试失败会明确报错，不会自动换 checkpoint 或替代数据集。

## 与真实 Full V16 汇总

先显式运行原 V16 的 native test（如尚无 best_psnr native 结果）；
不需要重新训练 V16。不要给汇总工具提供 PNG8 paper_metrics 结果或 V16-19 的验证结果。

```bash
python tools/qgrd_ablation_report.py \
  --dataset lsui \
  --baseline-run "<REAL_V16_LSUI_RUN_DIR>" \
  --no-qgrd-run "<A1_RUN_DIR>" \
  --shared-query-run "<A2_RUN_DIR>" \
  --single-stage-run "<A3_RUN_DIR>" \
  --no-local-refine-run "<A4_RUN_DIR>"
```

UIEB 用 --dataset uieb，并全部指定对应 UIEB 三集合实验目录。
工具只读取配置、run_info、冻结 TSV 和 native test CSV/JSON；不 import 模型、
不加载/修改 checkpoint，不评价图像，不修改实验目录。

必须一致：dataset、Train/Validation/Test hash、样本数量/IDs/filename、
metric protocol、evaluation resize/size/precision、best_psnr selector、seed、训练配方和公共模型配置。
同时检查 summary 与 CSV 的算术平均一致、结果完整且无重复 ID。
旧 V16 缺少 dataset 字段时只从 data.root/manifests 唯一推断，不默认猜 LSUI。
任何不匹配拒绝汇总并列出原因。

输出固定于：

```text
reports/qgrd_ablations/<lsui|uieb>/
  comparison.csv / comparison.json / comparison.md
  per_image_deltas.csv / protocol_audit.json
```

包含变体减 Full V16 的 ΔPSNR / ΔSSIM / ΔE00 和全部成对逐图差异。
未实现 bootstrap；不把单个 seed 的逐图差异解释为跨训练随机种子稳定性。
审计完成前不生成表；成功报告先在临时目录写完再替换旧报告，失败审计保留原报告。
不使用硬编码论文数字。

## 常见异常

- data.root 不存在：传 --data-root，确认上传真实 paired 图像；不修改冻结 TSV。
- source SHA256 不一致：部署同版本依赖文件；不要删 hash 校验或自动重新 pin 来掩盖变化。
- protocol hash/count 不一致：检查原 V16 三集合清单；不要替换成 V16-19 或重拆分。
- variant/version/config mismatch：使用本组自己的模块入口和 run/checkpoint，禁止跨组或跨 dataset resume。
- checkpoint epoch 与历史不同：使用最近完整保存的 last.pt；不要直接往已有较新历史追加旧 epoch。
- 报告拒绝比较：按具体 mismatch 复核 selector、native/PNG8 口径和冻结样本；不要删除审计断言。
- 测试缺少依赖：在测试环境安装所缺依赖后重试，不通过 skip/xfail 规避。
- 显存不足：第一轮受控实验不要只为某组调整 batch/patch；统一规划全部对照的新协议后再运行。

## 初始实现的本地验证记录

实际环境：Python 3.12.11、PyTorch 2.8.0、NumPy 2.2.6、
scikit-image 0.25.2、pytest 9.1.1；CUDA 不可用。
旧可视化全仓库检查首次因 Matplotlib 缺失而收集失败，使用项目外临时依赖目录
安装 Matplotlib 3.10.6 等依赖后重新运行通过；没有改动 requirements.txt 或全局环境。

最终实际结果：

- 新增 tests/qgrd_ablations：43 passed，0 failed，0 skipped。
- 要求的四个 regression 文件：62 passed，0 failed，0 skipped。
- 完整仓库 pytest：384 passed，0 failed，0 skipped。
- verify_all.sh：上述 43 + 62 项再次通过；五个 shell 脚本 bash -n 通过。
- 375 个原已跟踪文件 SHA256 全部不变，受保护路径 git diff 为空。
- 新增文件仅位于本任务指定的新目录/工具；无 pyc、__pycache__、DS_Store 或 IDE 变更进入交付。
- 标准配置与 V16 共有 state 逐项 torch.equal：A1 140、A2 309、A3 285、A4 261 个 tensor。
- 八组 LSUI/UIEB 风格临时三集合 smoke 的训练、验证、中断恢复、测试、strict checkpoint reload 均通过。

标准配置参数数量和共有 state 的 tensor-exact 初始化已实际检查。
新增测试覆盖模块矩阵、token 传递链、奇数尺寸、finite forward/backward、
无死参数、严格加载、跨组/跨数据集拒绝、八份配置、源码/清单篡改拒绝、
临时 LSUI/UIEB 合成三集合训练/验证/恢复/测试、连续与恢复权重/optimizer/RNG 精确一致，
以及只读报告的成功/拒绝场景和脚本 dry-run/失败停止。

工程 smoke 不是论文分数。本次没有启动正式 200 epoch GPU 训练，也没有批量复测真实 run；
CUDA AMP 下正式训练稳定性和性能增益须由用户在目标服务器实际验证。
