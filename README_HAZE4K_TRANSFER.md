# Haze4K：Full V16 与独立训练 w/o UICF

这是隔离的 **cross-task applicability** 实验：同一架构在 Haze4K 从随机初始化完整训练。两组均无 LSUI/UIEB 预训练、冻结、推理旁路或 `trans` 监督。代码不修改 `src/v15/`、`src/v16/`、`src/shared/`、旧配置、旧 split、旧实验。原有未跟踪 `haze4k_preflight.py` 也不属于此工程。

当前开发机没有 `/root/autodl-tmp/pro/publicdata/Haze4K`、没有 CUDA。**真实数据待 AutoDL 上执行**。合成测试数值仅验证软件，不是论文证据，不能填入真实结果表。没有完成真实预检或 200 epoch 训练的声明。

## 文件与调用关系

- `src/haze4k_transfer/prepare.py`：只读扫描、完整解码、命名配对、内容审计、分组划分、TSV/hash 固定。
- `dataset.py`：独立 Haze4K 清单；同步 crop/flip/rot90；V16 等价 padding；PIL RGB/BILINEAR 评价。
- `models.py`：Full 仅调用 `src.v16.models.network.build_model`；w/o UICF 仅调用 `src.v15.models.network.build_model`。两组 `model` 以完整 V16 原配置为来源，w/o 分支只保留 `in_channels,out_channels,base_channels,use_batch_norm,output_activation,color_query`，设 `type=plain_unet_color_query`。
- `train.py`：独立 AdamW/Charbonnier 训练、逐 epoch validation、best/last checkpoint、状态/RNG 恢复；不加载官方 test 图像。
- `eval.py`：strict checkpoint 加载、逐图 PSNR/SSIM、256/native 独立结果、固定样本图；复用 `src.v16.metrics.batch_metrics`。
- `compare.py`：配对身份及协议核对、clustered bootstrap、论文表、四列图。
- `utils.py`：可审计环境、原源码 hash、绘图与序列化；随机数和图像转换工具引用原 V16。
- `configs/haze4k/*.yaml`：相同协议，仅 `variant` 不同；`model` 与 `configs/config_v16.yaml:model` 完全相同，运行时强制检查。
- `scripts/haze4k/`：prepare、两组 train、eval/compare，以及完整 `run_all.sh`。
- `requirements-haze4k.txt`：仅此工程所需依赖；`tests/test_haze4k_transfer.py`：真实模型、合成数据软件测试。

未新增网络模块。不能修改新 YAML 的 model 参数来节省显存；工厂会拒绝。Full 在构建 UICF 前先构建骨干，因此分别重置 seed 后，Full.backbone 与 V15 独立模型全部 state_dict 张量完全一致。测试也覆盖参数差正好等于 UICF 参数量、奇数尺寸前向、有效反向。

## 安装与数据路径

以下假定仓库上传到 `/root/autodl-tmp/pro/UIE4_VN`；若实际位置不同，只修改 `cd` / `REPO`。在 **UIE4_VN 仓库根目录**运行。

```bash
cd /root/autodl-tmp/pro/UIE4_VN
# 优先保留 AutoDL 镜像已安装且适配驱动的 CUDA PyTorch。
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install -r requirements-haze4k.txt
python -m pytest tests/test_haze4k_transfer.py -q
```

Python >= 3.10、PyTorch >= 2.3。无 LPIPS、torchvision 或额外训练框架依赖。软件测试用 CPU；真实训练默认 GPU 0，无 CUDA 会明确退出，不会悄悄改用 CPU。依赖版本、设备、CUDA/cuDNN 与源码 hash 存入每次 run。

`--data-root` 可用于 prepare、train、eval；新 YAML 的 `data_root` 也可修改。prepare 的 `--split-dir` 和训练的 `--split-dir` 可共同覆盖默认 `split/haze4k`。相对工程路径统一相对于仓库根，manifest 中图像路径始终相对于 Haze4K 根，因此可以跨服务器搬迁。resume/eval 优先用 run 的清单副本，迁移后仅覆盖 `--data-root`，不改清单。

## 真实 prepare 是正式训练的前置条件

```bash
DATA_ROOT=/root/autodl-tmp/pro/publicdata/Haze4K
realpath "$DATA_ROOT"
ls -ld "$DATA_ROOT"/{train,test}/{haze,gt}
python -m src.haze4k_transfer.prepare --data-root "$DATA_ROOT" --seed 3520
```

预期的用户版本是官方 train 3000 haze / 3000 GT、test 1000 haze / 1000 GT。`train/haze/.DS_Store` 被报告且忽略，不删除。不会假设 750 GT × 4 雾版本。`trans/` 不扫描、不读取、不计算 hash、不进入训练或指标。

命名配对：`train/haze/1000_0.74_1.6.png -> train/gt/1000.png`；同样规则用于 test。GT stem 对应唯一常见图像扩展名；重复 stem（例如 `1.png` 和 `1.jpg`）、缺 GT、未使用 GT、损坏文件、尺寸不符都失败。目录可用唯一的 `hazy/clear` 别名，但与 `haze/gt` 同时存在会报错。图像目录中的意外子目录会报错。未知非图像文件报告具体相对路径；`.DS_Store`、`._*`、`Thumbs.db`、`desktop.ini` 明确标记为元数据。所有其他支持扩展名图像均完整解码，不按排序配对。

RGB 规则固定为 PIL `convert('RGB')`，不做 EXIF 旋转或 ICC 变换；记录原 mode，拒绝多帧图像，不允许截断解码。每对原始尺寸必须相同。

split 按 Unicode 字典序列出 clean ID，`random.Random(3520).shuffle`，前 `max(1, round_half_up(10%))` 个 clean ID 为 val，至少留下一个 train ID；清单按 input 相对路径字典序输出。当前版本如完整且每 GT 一 haze，结果 **2700 train / 300 validation / 1000 test**。多 haze 变体同 GT 永远只在一个 train/val 分组内，样本数以实际统计为准。官方 test 原样独立，不参与挑 checkpoint 或调参。

跨官方 split 数字 ID 重合是允许的。审计对 **所有输入/GT 角色**执行跨 train/test SHA256 和确定性 RGB 像素 hash 检查，因此也包含所要求的 GT/GT 检查；像素 hash 可发现文件编码不同但 RGB 完全相同的情况，不将其称为近似相似度。没有实现有阈值的近重复检索，报告不构成“排除了所有视觉近重复”的声明。

如发现内容重叠，程序保留官方划分，写 `overlap_review_required.json` 并退出，不生成可训练元数据。用户审阅后若决定仍保持官方划分，可显式重跑同命令加 `--accept-content-overlap`，元数据记录此决定；不可把该参数当成无条件跳过检查。目录、权限、解码异常均显示路径并失败。

产物：

```text
split/haze4k/
  train.tsv
  validation.tsv
  test.tsv
  inventory.json          # 每张 haze/GT 的文件 hash、RGB hash、尺寸、mode、官方 split/角色
  split_metadata.json     # 版本、算法、seed、counts、原 prepare 根路径、报告、各文件 SHA256
```

三个 TSV **包含表头**，固定四列：`sample_id\tinput_relative\tgt_relative\tclean_id`。示例：`train/1000_0.74_1.6`、`train/haze/1000_0.74_1.6.png`、`train/gt/1000.png`、`1000`。sample_id 含官方 split，跨官方 split 同 stem 不冲突。

重复 prepare 会重新扫描/解码和核对字节 hash，不重写已有文件。已有数据/seed/split 变化或缺文件会拒绝覆盖；需明确选择 **全新的** `--split-dir`。无自动重划和删除入口。每个 run 再保存整个 split 副本。训练/eval 检查元数据 hash 和对应数据文件 hash，数据新增/遗漏也会失败；训练仅读取 train/val 文件内容，test 仅检查目录清单。

## 训练、GPU smoke 与显存

主协议：seed 3520；200 epochs；batch 16；patch 256；AdamW lr 2e-4、betas 0.9/0.999、weight_decay 0；无 scheduler；Charbonnier epsilon 0.001；CUDA AMP；同步随机 crop/hflip/vflip/rot90；每 epoch validation；选择 validation PSNR 最优。小图在右/下 reflection padding，任一空间轴小于 2 时使用 edge，等价于 V16 数据实现。所有评价固定 float32、单图，避免两组 AMP 推理差异。

先在实际 GPU 上对两组运行两个 batch 的 smoke。验证仍用完整固定 validation，绝不加载 test：

```bash
python -m src.haze4k_transfer.train --config configs/haze4k/full_v16.yaml \
  --data-root "$DATA_ROOT" --gpu 0 --smoke-epochs 1 --smoke-steps 2
python -m src.haze4k_transfer.train --config configs/haze4k/wo_uicf_v15_backbone.yaml \
  --data-root "$DATA_ROOT" --gpu 0 --smoke-epochs 1 --smoke-steps 2
```

smoke 独立 run，`smoke=true`，结果表明确标为不可用于论文；不接着把其 checkpoint 当主训练初始化。日志记录实际 epoch 秒数；短 smoke 仅确认显存/流水线，主训练第一个完整 epoch 时间更能判断 200 epochs 工期。

发生 OOM 会报错退出，绝不会自动改输入尺寸或 batch。若需调整，**两组同时**使用例如 `--batch-size 8 --accumulation-steps 2`，先对两组重做同配置 smoke，再对两组从头主训练，effective batch 仍为 16。显存不足的失败 run 保留失败状态。梯度累积按 group 实际样本数归一化，最后不足组如实较小。BatchNorm 的 batch 统计随物理 batch 改变，累积不等价于真实 batch16，必须在论文协议中记录；比较会拒绝两组 batch/accumulation 等不一致的结果。

```bash
python -m src.haze4k_transfer.train --config configs/haze4k/full_v16.yaml --gpu 0 --data-root "$DATA_ROOT"
python -m src.haze4k_transfer.train --config configs/haze4k/wo_uicf_v15_backbone.yaml --gpu 0 --data-root "$DATA_ROOT"
```

默认 workers=4。可两组一致传 `--num-workers 0`；same seed 不意味着逐步增强相同，尤其单进程增强 RNG 会受模型初始化消耗影响。数据排序/采样 generator 有独立 seed；完整运行顺序、seed、worker/AMP 设置都记录，不用不同超参偏向 Full。

每个运行输出一行真实 `RUN_DIR=...`。不加载任何外部权重。时间戳使用 UTC 加微秒及随机后缀，`exist_ok=False` 防覆盖。

```text
experiments/haze4k_transfer/<variant>_<timestamp>_3520_<suffix>/
  config_source.yaml      # 配置原文
  config_resolved.yaml    # CLI 覆盖后的配置和 synthetic/smoke 标记
  environment.json        # git commit/dirty、包版本、命令、设备、源码 hash
  model.json              # 原工厂、实际参数量、随机初始化声明
  runtime.json            # 当前数据根、实际 AMP、effective batch、恢复信息
  split/                  # 固定清单、元数据和 inventory 副本
  checkpoint/last.pt
  checkpoint/best_psnr.pt
  status.json
  train.log
  history.json
  history.csv
  curves.png              # train/val loss、val PSNR/SSIM
  validation/epoch_0001.csv ...
```

NaN/Inf 输出、loss、梯度、参数、指标和坏图都失败，绝不跳过后继续求均值。原指标定义保留完全相同图像的 PSNR=Inf；此工程遇此情况也明确失败并要求调查，而非输出非标准 JSON/置信区间。

## 重启恢复

```bash
# FULL_RUN 必须取该运行实际打印的绝对路径。
python -m src.haze4k_transfer.train --resume "$FULL_RUN/checkpoint/last.pt" --gpu 0 --data-root "$DATA_ROOT"
```

只允许恢复 `last.pt`，避免从旧 best 分叉覆盖同 run 历史。恢复 model(strict)、AdamW、AMP scaler、epoch、best validation PSNR、Python/NumPy/Torch/全部 CUDA RNG、训练和验证 DataLoader generator、历史。仅支持 **epoch 边界**恢复，中途失败回退到上一个 last epoch。必须相同源码 hash/保存配置/清单；迁移数据可覆盖根目录，不能在 resume 改 batch/seed/模型/目标 epoch。`--stop-after-epoch 1` 提供可控中断，保留原目标 epoch，用于测试恢复。

workers 非持久，每轮重建；同环境/设备/workers 且确定性 kernel 下尽可能复现。默认原 V16 协议 `deterministic=false`，CUDA 非确定性、不同设备/库版本和并发调度可能造成差异，不能承诺逐 batch 位级复现。测试实际检查了 CPU workers0 确定性模式中恢复两轮与连续两轮模型权重逐项相等。CPU 不支持 CUDA AMP，因此 CPU 测试仅覆盖禁用 scaler 的保存/恢复；启用 AMP 的状态恢复需真实 CUDA smoke 验证。

## 独立 eval 与 compare

训练结束后，把实际打印的两条 `RUN_DIR` 赋给变量（以下不是虚构 run 名）：

```bash
read -r -p '粘贴 Full 的实际 RUN_DIR: ' FULL_RUN
read -r -p '粘贴 w/o UICF 的实际 RUN_DIR: ' WO_RUN
python -m src.haze4k_transfer.eval --run-dir "$FULL_RUN" --checkpoint best_psnr --gpu 0 --resolution 256 --data-root "$DATA_ROOT"
python -m src.haze4k_transfer.eval --run-dir "$WO_RUN" --checkpoint best_psnr --gpu 0 --resolution 256 --data-root "$DATA_ROOT"
python -m src.haze4k_transfer.compare --full-run "$FULL_RUN" --wo-run "$WO_RUN" --resolution 256
# 或 GPU=0 HAZE4K_ROOT="$DATA_ROOT" bash scripts/haze4k/eval_compare.sh "$FULL_RUN" "$WO_RUN"
```

`test_256` 严格为 PIL BILINEAR 将 input/GT 各自缩放至 256×256，float RGB [0,1]，prediction clamp [0,1]，每图原 V16 PSNR 与 Gaussian SSIM：data_range=1、crop_border=0、window=11、sigma=1.5。总指标是 **逐图指标算术平均**，不是先全局平均 MSE 再转 PSNR。验证采用完全相同协议，以其 PSNR 选 best。可 `--split validation` 独立核对验证结果。test 只用于最终报告。程序要求目标轮数已完成，且 best checkpoint 对应完整验证历史中的最高 PSNR；中断运行只能独立评价 validation，不能提前生成正式 test。

两次 eval 各生成 `test_256/summary.json`、`per_image.csv`（四身份列+psnr/ssim）、预先固定 IDs 的 images 及 selection JSON。默认取官方 test sample_id 字典序前 8 个，选择不依赖预测效果；可 `--save-all` 保存全部测试输出。PNG 是 8-bit 展示图，不用于指标；指标来自 float32 tensor。已有 eval 输出拒绝覆盖；同 checkpoint 和协议仅执行一次即可。

compare 先检查模型类型、checkpoint=best_psnr、模型/训练配置、实际 AMP、源码/指标版本、全部 split/hash、官方 test sample_id/input/GT/clean ID、分辨率与均值一致性。只有逐图完全对齐才统计 Full−w/o 的 Δ、正增益比例；默认 5000 次、seed3520 的 paired clean-ID clustered percentile bootstrap 95% CI。每个重采样 clean ID 包含其全部 haze 版本，计算 image-weighted mean；当前一 GT 一 haze 就退化为普通配对 bootstrap。可改 `--bootstrap-iterations`（至少2000）和 `--bootstrap-seed`，记录实际值。

```text
<FULL_RUN>/comparisons/<WO_RUN 的目录名>_test_256/
  paper_table.md
  summary.json
  paired_deltas.csv
  visualization_selection.json
  panels/<固定 sample_id 的 SHA256 前20位>.png  # Hazy | w/o UICF | Full | GT
```

原生分辨率是额外独立协议：分别 eval `--resolution native` 后 compare `--resolution native`，写 `test_native`。不做外部 pad/tile/resize，仅复用原网络已有内部 pad/crop；软件测试覆盖奇数尺寸兼容性。真实高分辨率显存仍待 AutoDL 验证，OOM 明确失败，不会偷偷 resize 后标 native。256 与 native 不能混表。

论文只能支持新任务从头训练的适用性和 UICF 在完整框架上的增量贡献；不支持零样本迁移、all-in-one、全领域 SOTA 或 QGRD 独立跨任务效用。2700训练/300验证与公开全部3000训练对照不严格等价，必须注明 resize/native 的区别。

## 软件测试

本机实际结果：12 passed、1 skipped（无 CUDA，跳过启用 AMP 的恢复测试）。完整交付验证记录见 [HAZE4K_VALIDATION.md](HAZE4K_VALIDATION.md)。AutoDL 有 CUDA 时该测试会实际运行。

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest tests/test_haze4k_transfer.py -q --durations=10
```

临时目录构造合成数据，包括完整 3000/1000 小图模拟；验证配对、损坏/尺寸/歧义/缺失/重复/覆盖防护、同 ID 分组、2700/300/1000、确定性文件 hash、元数据保留、trans 不读取、真实模型同源性及前后向、两组各两轮训练与恢复、连续/恢复一致、256和native评估、拒绝不一致比较、bootstrap、PNG四列和出表。大型 checkpoint 在测试 finally 中仅清理本测试新建的 run，不接触旧实验。

## AutoDL 一次复制操作手册

下面会先确认路径和 CUDA、安装依赖及测试，再执行真实 prepare → 两组 GPU smoke → Full 200 epochs → w/o UICF 200 epochs → 两组 eval → compare。仅当实际预检及前一步成功才继续；实际 run-dir 从成功训练日志中自动读取，无伪造路径。长任务可在 tmux 中运行。

```bash
set -euo pipefail
REPO=/root/autodl-tmp/pro/UIE4_VN
export HAZE4K_ROOT=/root/autodl-tmp/pro/publicdata/Haze4K
export GPU=0
cd "$REPO"
realpath "$HAZE4K_ROOT"
ls -ld "$HAZE4K_ROOT"/{train,test}/{haze,gt}
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
python -m pip install -r requirements-haze4k.txt
OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest tests/test_haze4k_transfer.py -q
# 若 smoke OOM，明确两组改为相同 BATCH_SIZE=8 ACCUMULATION_STEPS=2 后重跑本脚本；不会自动调整。
export BATCH_SIZE=16
export ACCUMULATION_STEPS=1
bash scripts/haze4k/run_all.sh
```

`run_all.sh` 重跑会新建两组 run，适合新的成对实验；中断已有训练应使用上文 `--resume`，完成后手工 eval/compare，避免重复200轮。主训练可顺序跑同一 GPU；不要用官方 test 的效果决定是否重训、选 epoch 或调整超参。
