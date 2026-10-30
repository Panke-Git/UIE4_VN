# ICAR-UIE V16 校正参数化消融：Mac 开发与 WSL 运行指南

初始代码交付只在 MacBook 的 `UIE4_workspace/UIE4_VN` 开发、进行轻量 CPU 验证。后续于 2026-10-31 获得授权，通过 SSH 对 WSL 数据目录和清单做了只读核验，确认本次原 V16 协议应使用 `/home/paxton/pro/publicdata/LSUI19_dup_train`。没有启动正式训练或评估。以下 WSL/Git 命令仍供你手动执行。

## 研究范围、实现与容量

唯一正式基线是：

```text
/home/paxton/pro/UIE4_VN/experiments/v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528
```

使用该运行的 `best/best_psnr.pt` 和原生 `result/test_summary.json`；不重新训练 Full UICF，不改用 V16-19、Haze4K、PNG8 paper metrics 或验证集结果。用户提供的 Full 测试值为 PSNR=28.388665、SSIM=0.928637、ΔE00=5.204765。后续目录检查已读取基线配置和冻结清单，但没有重新核验测试摘要或权重内容；指标及其与部署代码的一致性仍标记为**待 WSL 验证**。

| 标识 | 最终校正公式 | 系数/残差语义 |
|---|---|---|
| `direct_residual` | `I_c = I + predicted_residual` | MLP 直接预测 RGB 残差；诊断的 `correction_field=None` |
| `without_anchor` | `I_c = I + R * I` | 保留系数预测、局部/坐标/全局条件，等价于 b=0 |
| `fixed_anchor` | `I_c = I + R * (I - 0.5)` | RGB 三通道固定 0.5，不是训练集均值 |
| 原始 Full UICF | `I_c = I + R * (I - b)` | 原始图像自适应锚点；本轮不改动 |

实现位于独立包 `src/uicf_parameterization_ablations/`。工厂直接复用 V15 Backbone（含完整 Color Query/QGRD），先建 Backbone，再按原 UICF 构造顺序建 Encoder、坐标编码、Anchor Predictor、CorrectionFieldMLP。原 Anchor Predictor 完成初始化消耗后被物理移除，替换成**无参数**的原样全局平均池化接口；固定变体继续调用原 `use_learned_anchor=False` 的 forward。未添加补偿层、注意力、门控、激活或损失，未冻结主要模块，也不靠运行时修改原 V16 工厂。

保留原构造顺序的目的是让同一 seed 的所有共有参数和 BatchNorm 缓冲区逐项一致，而非保留无效参数凑容量。标准维度模型已在本地用真实 autograd 输出图检查：

| 模型 | 注册参数 | 可训练参数 | 参与输出前向 | 不参与输出 | 相对 Full |
|---|---:|---:|---:|---:|---:|
| Full UICF | 38,878,217 | 38,878,217 | 38,878,217 | 0 | 0 |
| 三个新变体（各） | 38,874,886 | 38,874,886 | 38,874,886 | 0 | −3,331 |

三组与 Full **不是严格参数匹配**。删除的 3,331 个参数均来自 Anchor MLP。输出层零初始化保证初始 UICF `I_c=I`；初始化时上游梯度可以为零，但输出图仍连通，测试在更新一次输出层后再次检查非零梯度。研究结论应讨论经验表现，不应宣称 anchor-relative 形式数学表达能力必然更强。

## 第一阶段：MacBook 本地检查

保持 Codex 当前 `UIE4_workspace` 工作区不变，只在终端进入其子工程：

```bash
cd ~/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN
git status --short --branch
git diff --check
git diff -- src/v16 src/v15 src/shared/uicf_inr.py src/shared/uicf_models.py configs/config_v16.yaml
find src/uicf_parameterization_ablations configs/ablations/uicf_parameterization \
  scripts/uicf_parameterization_ablations tests/uicf_parameterization_ablations \
  -type f ! -path '*/__pycache__/*' | sort
sed -n '1,240p' src/uicf_parameterization_ablations/models/network.py
```

本机已实际验证的解释器是 `/opt/anaconda3/envs/Pytorch/bin/python`。系统 `python3` 缺少 torch，Codex 当前捆绑 Python 也缺少相关测试依赖；不要把它们的检查结果当作本项目环境。

```bash
PYTHON=/opt/anaconda3/envs/Pytorch/bin/python bash scripts/uicf_parameterization_ablations/verify_local.sh

# 独立 CLI 检查：三组人工 2/1/1 数据、微型模型、CPU、2 epoch；
# 每组在第 1 epoch 保存后中断，再恢复并显式测试，不是正式实验。
/opt/anaconda3/envs/Pytorch/bin/python -m src.uicf_parameterization_ablations.smoke

# 标准维度的轻量参数/输出图检查：随机初始化、32×32 人工输入，无优化器更新。
/opt/anaconda3/envs/Pytorch/bin/python -m src.uicf_parameterization_ablations.capacity

# 只检查正式配置/清单哈希与 CLI，不建模型、不访问真实图像、不启动训练。
/opt/anaconda3/envs/Pytorch/bin/python -m src.uicf_parameterization_ablations.train \
  --config configs/ablations/uicf_parameterization/direct_residual.yaml \
  --variant direct_residual --seed 3520 --gpu 0 --dry-run
```

检查本次开发开始时全部已跟踪文件是否仍逐字节不变：

```bash
/opt/anaconda3/envs/Pytorch/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path.cwd()
snapshot = json.loads((root/'src/uicf_parameterization_ablations/protection_snapshot.json').read_text())
changed = [p for p,h in snapshot['tracked_files'].items()
           if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest() != h]
assert not changed, changed
print('原有文件全部不变：', len(snapshot['tracked_files']))
PY
```

`analysis/` 和实验输出受原 `.gitignore` 排除；不要将人工 smoke 的图像、checkpoint、指标复制到正式实验。测试明细见同目录的 `UICF_PARAMETERIZATION_LOCAL_TEST_REPORT_CN.md`。

## 第二阶段：手动 GitHub 同步

### Mac：人工检查、暂存与提交

本次开发开始时 `main` 比已有的 `origin/main` 跟踪引用落后 1 个提交；没有执行 fetch/pull，引用也可能已经过时。以下步骤由你人工完成，不能直接忽略分支分歧。

```bash
cd ~/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN
git status --short --branch
git diff
git fetch origin
git log --oneline --left-right HEAD...origin/main
```

若仍然只是本地落后、已跟踪文件干净且确认上游不会与新增文件冲突，可手动 `git merge --ff-only origin/main`。Git 拒绝快进或提示文件冲突时停下并人工审查；不使用 reset、clean、强制 checkout 或强制推送。本次保护快照记录的是开发起点，之后人工合并上游对旧文件的修改会导致快照测试报警，应先审查来源，不自动重新生成快照掩盖差异。

确认内容和测试后，选择性加入本次文件：

```bash
git add src/uicf_parameterization_ablations \
  configs/ablations/uicf_parameterization \
  scripts/uicf_parameterization_ablations \
  tests/uicf_parameterization_ablations \
  tools/uicf_parameterization_report.py \
  docs/UICF_PARAMETERIZATION_ABLATION_GUIDE_CN.md \
  docs/UICF_PARAMETERIZATION_LOCAL_TEST_REPORT_CN.md \
  docs/UICF_PARAMETERIZATION_GIT_REPORT_CN.md
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Add isolated V16 UICF correction parameterization ablations"
git push origin main
```

如 push 被拒绝，先 fetch、检查分歧并人工合并；不要使用 `--force`。Codex 没有替你执行这些命令。

### WSL：保留未提交工作，安全同步

以下在你自己的 WSL Bash 终端执行：

```bash
cd /home/paxton/pro/UIE4_VN
git status --short --branch
git branch --show-current
git diff
git diff --cached
```

只有确认分支是 `main`、没有未提交/未跟踪的代码需要保留时，再执行：

```bash
if [ "$(git branch --show-current)" != main ] || [ -n "$(git status --porcelain)" ]; then
  printf '%s\n' '停止同步：请先人工处理分支和未提交工作；不要覆盖本地代码。'
else
  git pull --ff-only origin main
fi
git status --short --branch
```

本指南不自动 stash、不覆盖 WSL 工作树，也不改变忽略的原有实验和 checkpoint。

## 第三阶段：WSL 环境、GPU、数据与基线检查

先激活你实际安装 PyTorch 的环境。数据根目录已核实为 `/home/paxton/pro/publicdata/LSUI19_dup_train`；WSL 环境名称、Python/PyTorch/CUDA 版本、GPU 型号与显存仍需你手动检查。

```bash
cd /home/paxton/pro/UIE4_VN
pwd
git status --short --branch
command -v python
python --version
nvidia-smi
python - <<'PY'
import sys, torch, numpy, PIL, yaml, skimage, pytest
print('Python:', sys.executable, sys.version)
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('numpy/Pillow/PyYAML/skimage/pytest:', numpy.__version__, PIL.__version__, yaml.__version__, skimage.__version__, pytest.__version__)
print('CUDA available:', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print('GPU', i, p.name, 'total GiB=', p.total_memory/2**30)
assert torch.cuda.is_available(), '请选择正确 CUDA 环境，暂勿启动正式实验'
PY
```

依赖复用已有 `requirements.txt`，新增功能没有引入额外 Python 库。若缺包，先在你选择的环境安装匹配硬件的 PyTorch，以及 `python -m pip install -r requirements.txt`；原 requirements 不安装 PyTorch，不能用它替代 CUDA/PyTorch 安装。随后再检查版本。不要自动更换全局环境。

数据根目录已经确认。按原 V16 清单逐项检查，`LSUI19_dup_train` 的训练 3466 对、验证 385 对、测试 428 对文件全部存在，当前三份清单与指定基线冻结清单的 SHA256 全部一致。`LSUI19` 的 Train/Val 文件分布与该协议不同，请使用下面已经填好的路径，不修改固定划分。

```bash
ls -ld /home/paxton/pro/UIE4_VN /home/paxton/pro/publicdata
ls -l split/lsui19/{train,validation,test}.tsv
export LSUI_ROOT=/home/paxton/pro/publicdata/LSUI19_dup_train
test -d "$LSUI_ROOT"
export UICF_GPU=0   # 根据 nvidia-smi 人工选择，0 只是示例 GPU 索引
python - <<'PY'
import os
from pathlib import Path
from src.v16.dataset import validate_split_protocol, verify_image_entries
from src.uicf_parameterization_ablations.protocol import SPLITS, manifest_hashes, config_path
from src.v16.utils import load_yaml
paths = {s: Path('split/lsui19')/(s+'.tsv') for s in SPLITS}
cfg = load_yaml(config_path('direct_residual'))
entries = validate_split_protocol(paths, cfg['data']['expected_counts'])
print('固定清单 SHA256:', manifest_hashes(cfg, paths))
root = Path(os.environ['LSUI_ROOT'])
for split, rows in entries.items():
    verify_image_entries(rows, root, split)
    print(split, len(rows), '图像配对检查通过')
PY
```

文件存在性与清单哈希检查已经完成；上面的环境检查、PIL 解码和逐图配对内容检查仍由你后续手动运行，不能将文件存在等同于图像内容已验证。正式训练入口只读取测试清单以验证划分，不载入测试图像；图像完整性预检和显式 test 才访问测试图。

检查唯一基线，**只读取元数据并 stat checkpoint，不加载权重**：

```bash
export FULL_RUN=/home/paxton/pro/UIE4_VN/experiments/v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528
python - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ['FULL_RUN'])
for rel in ['config_resolved.yaml','run_info.json','status.json','best/best_psnr.json','best/best_psnr.pt','result/test_metrics.csv','result/test_summary.json']:
    p = r/rel
    assert p.is_file(), p
    print(p, p.stat().st_size)
s = json.loads((r/'result/test_summary.json').read_text())
print('基线原生测试摘要:', s)
assert s['sample_count'] == 428
assert Path(s['checkpoint']).parts[-2:] == ('best','best_psnr.pt')
PY
python -m src.uicf_parameterization_ablations.train --variant direct_residual --seed 3520 --gpu "$UICF_GPU" --data-root "$LSUI_ROOT" --dry-run
python -m src.uicf_parameterization_ablations.train --variant without_anchor --seed 3520 --gpu "$UICF_GPU" --data-root "$LSUI_ROOT" --dry-run
python -m src.uicf_parameterization_ablations.train --variant fixed_anchor --seed 3520 --gpu "$UICF_GPU" --data-root "$LSUI_ROOT" --dry-run
bash scripts/uicf_parameterization_ablations/verify_local.sh
```

### Batch Size 16 资源预检（WSL 上人工执行）

`nvidia-smi` 不能证明训练能装入显存。下面依次对标准模型做一个 **batch=16、256×256、AMP、AdamW** 的人工数据优化步骤，不载入原 checkpoint，不保存权重，不是正式实验。它测量完整前后向和优化器状态分配；实际长程训练与数据加载仍需观察。此命令本轮没有执行。

```bash
python - <<'PY'
import gc, os, torch
from src.v16.train import _grad_scaler
from src.v16.losses import build_loss
from src.v16.utils import load_yaml
from src.uicf_parameterization_ablations import VARIANTS
from src.uicf_parameterization_ablations.protocol import config_path
from src.uicf_parameterization_ablations.models import build_model
device = torch.device('cuda', int(os.environ['UICF_GPU']))
torch.cuda.set_device(device)
for variant in VARIANTS:
    torch.manual_seed(3520)
    cfg = load_yaml(config_path(variant))
    model = build_model(cfg['model']).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0, betas=(0.9,0.999))
    scaler = _grad_scaler(True)
    criterion = build_loss(cfg['loss']).to(device)
    x, y = torch.rand(16,3,256,256,device=device), torch.rand(16,3,256,256,device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast('cuda', enabled=True):
        pred = model(x)
        loss = criterion(pred, y)
    assert torch.isfinite(pred).all() and torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize(device)
    print(variant, '单步通过；peak allocated GiB=', torch.cuda.max_memory_allocated(device)/2**30)
    del loss, pred, x, y, criterion, scaler, opt, model
    gc.collect()
    torch.cuda.empty_cache()
PY
```

OOM 或数值异常时停止，不自动降低 batch、模型规模、分辨率，不改变 BatchNorm，不把梯度累积视为原 batch 16 的严格等价替代。本轮无法确认你的 WSL GPU 能否运行原协议。

## 第四阶段：WSL 正式训练（严格串行）

下列每条均从随机初始化训练完整网络，200 epoch、batch 16、seed 3520、AdamW lr=0.0002 / wd=0、Charbonnier eps=0.001、AMP=True，每 epoch 验证，用最佳验证 PSNR 保存主 checkpoint。`--seed` 可显式指定，但本受控配置拒绝非 3520；避免无意增加随机种子研究变量。不要附加 Full UICF 权重初始化。

```bash
cd /home/paxton/pro/UIE4_VN
export LSUI_ROOT=/home/paxton/pro/publicdata/LSUI19_dup_train
test -d "$LSUI_ROOT"
: "${UICF_GPU:?先选择 GPU}"

python -m src.uicf_parameterization_ablations.train \
  --config configs/ablations/uicf_parameterization/direct_residual.yaml \
  --variant direct_residual --seed 3520 --gpu "$UICF_GPU" \
  --data-root "$LSUI_ROOT" --output-dir experiments/uicf_parameterization/direct_residual

python -m src.uicf_parameterization_ablations.train \
  --config configs/ablations/uicf_parameterization/without_anchor.yaml \
  --variant without_anchor --seed 3520 --gpu "$UICF_GPU" \
  --data-root "$LSUI_ROOT" --output-dir experiments/uicf_parameterization/without_anchor

python -m src.uicf_parameterization_ablations.train \
  --config configs/ablations/uicf_parameterization/fixed_anchor.yaml \
  --variant fixed_anchor --seed 3520 --gpu "$UICF_GPU" \
  --data-root "$LSUI_ROOT" --output-dir experiments/uicf_parameterization/fixed_anchor
```

也可在完成第三阶段后，手动调用 `bash scripts/uicf_parameterization_ablations/train_sequential.sh "$LSUI_ROOT" "$UICF_GPU"`。该脚本遇错停止，不自动改配置、不自动测试。每个训练入口输出 `RUN_DIR=...`；记录三个准确路径，不自动猜测最新目录。

### 断点恢复

使用该变体自己的最近完整 `checkpoint/last.pt`；不能拿最佳 checkpoint 向已有较新历史追加，也不能跨变体恢复。下面的路径通过你实际粘贴设定：

```bash
read -r -p '粘贴 Direct Residual 的 RUN_DIR: ' DR_RUN
read -r -p '粘贴 w/o Anchor 的 RUN_DIR: ' WA_RUN
read -r -p '粘贴 Fixed Anchor 的 RUN_DIR: ' FA_RUN
export DR_RUN WA_RUN FA_RUN

# 仅对中断、未到 200 epoch 的对应运行执行，不要依次恢复已完成运行。
python -m src.uicf_parameterization_ablations.train --resume "$DR_RUN/checkpoint/last.pt" --variant direct_residual --gpu "$UICF_GPU" --data-root "$LSUI_ROOT"
python -m src.uicf_parameterization_ablations.train --resume "$WA_RUN/checkpoint/last.pt" --variant without_anchor --gpu "$UICF_GPU" --data-root "$LSUI_ROOT"
python -m src.uicf_parameterization_ablations.train --resume "$FA_RUN/checkpoint/last.pt" --variant fixed_anchor --gpu "$UICF_GPU" --data-root "$LSUI_ROOT"
```

恢复核对 version、variant、公式、完整配置、seed、源码哈希、冻结清单、checkpoint epoch 与历史末轮，再严格加载模型、优化器、AMP、Python/NumPy/Torch/CUDA RNG 和 DataLoader generator。仅允许 `--data-root` 迁移同一批图像，不允许改 recipe。`--stop-after-epoch N` 只用于完整保存某轮后中断，不修改配置的总训练轮数。精确恢复以 epoch 边界为单位；相同软件、设备、worker 数和确定性设置是可重复条件，不承诺恢复尚未保存的半轮，也不承诺 CUDA 非确定性算法位级一致。

输出包括：

```text
<RUN_DIR>/config_source.yaml / config_resolved.yaml / config.json
          run_info.json / status.json
          split_snapshot/train.tsv / validation.tsv / test.tsv
          best/best_psnr.pt / best_psnr.json (+ best_ssim / best_loss)
          checkpoint/last.pt / epoch_*.pt
          log/train.log / metrics_history.csv / metrics_history.json
          result/validation_metrics.csv / validation_summary.json
```

新 checkpoint 还记录 variant、formula、resolved_config、seed、epoch/训练状态、三集合哈希、代码版本/源码哈希、容量、model/optimizer/AMP/RNG/generator 状态。不会改变原 V16 checkpoint schema。

## 第五阶段：WSL 正式测试

只有训练完成后才允许显式 test。先记录准确 RUN_DIR（如还未设置，使用上一节三个 `read` 命令），再检查：

```bash
python - <<'PY'
import json, os
from pathlib import Path
for variable in ['DR_RUN','WA_RUN','FA_RUN']:
    r = Path(os.environ[variable])
    status = json.loads((r/'status.json').read_text())
    assert status['status'] == 'completed' and status['last_epoch'] == 200, (r,status)
    p = r/'best/best_psnr.pt'
    assert p.is_file() and p.stat().st_size > 0, p
    print(r, status, json.loads((r/'best/best_psnr.json').read_text()))
PY

python -m src.uicf_parameterization_ablations.test --run-dir "$DR_RUN" --variant direct_residual --checkpoint best_psnr --gpu "$UICF_GPU" --data-root "$LSUI_ROOT"
python -m src.uicf_parameterization_ablations.test --run-dir "$WA_RUN" --variant without_anchor --checkpoint best_psnr --gpu "$UICF_GPU" --data-root "$LSUI_ROOT"
python -m src.uicf_parameterization_ablations.test --run-dir "$FA_RUN" --variant fixed_anchor --checkpoint best_psnr --gpu "$UICF_GPU" --data-root "$LSUI_ROOT"
```

也可手动 `bash scripts/uicf_parameterization_ablations/test_runs.sh "$LSUI_ROOT" "$UICF_GPU" "$DR_RUN" "$WA_RUN" "$FA_RUN"`。测试从冻结配置构造对应变体，`--variant` 只是额外身份断言，传错会报错。入口仅接受 `best_psnr`，不会按测试集表现选择 checkpoint。

评估完全使用原 V16 数据集和度量函数：PIL bilinear resize 到 256×256、CUDA 环境 AMP forward、预测转 float32 并 clamp [0,1]、**PNG 编码前**计算 PSNR/SSIM/ΔE00、每图等权均值。SSIM window=11 / sigma=1.5 / border=0；ΔE00 用原 `src/shared/e00.py` 的 CIEDE2000、D65/2°、kL=kC=kH=1，逐像素后逐图平均。测试 batch size=1；没有变更训练 batch 16。

结果保存为 `result/test_metrics.csv`、`result/test_summary.json`，并保存增强图、样例和网格。CSV 保留 filename/sample_id/psnr/ssim/e00，可用于之后配对统计；summary 含 variant、checkpoint、原生 float 指标口径、数量和划分哈希。再次测试会更新本运行的同名测试输出；需要保留之前输出时先人工备份，不混用不同口径结果。

检查完整性：

```bash
python - <<'PY'
import csv, json, os, math
from pathlib import Path
for variable in ['DR_RUN','WA_RUN','FA_RUN']:
    r = Path(os.environ[variable])
    s = json.loads((r/'result/test_summary.json').read_text())
    with (r/'result/test_metrics.csv').open() as f: rows = list(csv.DictReader(f))
    assert s['sample_count'] == len(rows) == len({x['sample_id'] for x in rows}) == 428
    assert s['checkpoint_selector'] == 'best_psnr' and s['split'] == 'test'
    assert not s['smoke_test']
    for metric in ['psnr','ssim','e00']:
        values = [float(x[metric]) for x in rows]
        assert all(math.isfinite(x) for x in values)
        assert abs(sum(values)/428-s['mean_'+metric]) < 1e-6
    print(r, s['mean_psnr'], s['mean_ssim'], s['mean_e00'])
PY
```

## 第六阶段：正式结果汇总

参数量文件可在 WSL 用相同代码重新生成，它只用随机模型和小 CPU 输入检查输出图，不是训练、测试或 checkpoint 推理：

```bash
python -m src.uicf_parameterization_ablations.capacity
python tools/uicf_parameterization_report.py \
  --baseline-run "$FULL_RUN" \
  --direct-residual-run "$DR_RUN" \
  --without-anchor-run "$WA_RUN" \
  --fixed-anchor-run "$FA_RUN" \
  --parameter-counts analysis/uicf_parameterization/parameter_counts.json \
  --output-dir analysis/uicf_parameterization
```

生成 `analysis/uicf_parameterization/results.md` 和 `results.csv`。汇总只读元数据、固定 TSV、逐图 CSV 和文件存在性，不加载模型或权重、不重新评估。它优先读取指定原 V16 的真实 `test_summary.json`，校验唯一基线目录名、版本、原训练 recipe、seed、样本数量/ID/filename、三集合哈希、best_psnr、原生 float 指标口径和 CIEDE2000 协议，并核对逐图均值。不接受 smoke、V16-19 验证集或不匹配的实验。

未完成某组时可以省略该组 `--...-run`，对应指标保持缺失；已提供但身份/协议冲突时直接报错，不悄悄改用验证指标或新猜一个运行。表中同时提供总/可训练/有效/无效参数，以及 Variant−Full 的 PSNR、SSIM、ΔE00 差值；负的 ΔE00 差值表示改善。

Mac 没有基线文件时，仅为生成缺失值模板，显式执行：

```bash
/opt/anaconda3/envs/Pytorch/bin/python tools/uicf_parameterization_report.py --use-provided-baseline
```

此时 Full 一行明确标为“用户提供，待 WSL 验证”；三个新变体全部是缺失值。不要将它解释为已经完成的消融。正式 WSL 汇总不需要也不建议使用此回退标志。

## 尚待目标环境确认的事项

- WSL 当前代码是否包含本次交付、其 Git 状态和源码版本；本轮没有远程读取。
- 数据根目录及全部清单路径的文件存在性已经通过 SSH 确认；逐图解码、配对内容及其与历史基线数据的内容一致性仍需验证。
- 指定 Full UICF 文件、测试摘要、best_psnr 的原始选择信息及其与当前源代码的一致性。
- CUDA AMP 的运行稳定性、batch 16 显存容量、200 epoch 正式训练和 428 张真实测试指标。
- 本地已经执行的检查、初次测试目录设置问题和最终结果见本地测试报告；不将未执行的 WSL 检查计为通过。
