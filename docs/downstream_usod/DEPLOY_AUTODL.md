# AutoDL 后续部署与运行

目标机通过 ssh school 访问。当前的主项目工作树位于 /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN，正式结果也保存在这个目录的 experiments 下。部署脚本将发布分支检出至同级 UIE4_VN-usod，检查 origin、分支、提交及工作树状态。如果指定目标已存在，只有干净且完全相同的检出能通过检查。

## 获取发布版本

从 Mac 的本功能 worktree 执行。release 使用实际已推送的完整 SHA。

```bash
release=$(git rev-parse HEAD)
git ls-remote origin refs/heads/feat/downstream-usod-tc-usod
ssh school "bash -s -- --commit $release --checkout /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN-usod" \
  < scripts/downstream_usod/deploy_autodl.sh
```

部署脚本按 Linux 编写，采用独立临时目录和无覆盖移动。现有 AutoDL 主工作树已有修改，使用新目录即可保留它。

## 迁移正式资产

Mac 已具备官方 Test 三个模态和完整 TC-USOD checkpoint。以下是后续从 Mac 迁入新实例的命令。

```bash
workspace=/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace
ssh school 'mkdir -p /root/autodl-tmp/pro/publicdata/USOD10K /root/autodl-tmp/pro/publicdata/USOD10K_assets'
rsync -a --partial "$workspace/publicdata/USOD10K/USOD10K-TE" \
  school:/root/autodl-tmp/pro/publicdata/USOD10K/
rsync -a --partial "$workspace/publicdata/USOD10K_assets/TCUSOD.pth" \
  school:/root/autodl-tmp/pro/publicdata/USOD10K_assets/
```

从已有可信训练存档将 LSUI 正式 run 的 config_resolved.yaml 和 best/best_psnr.pt 复制到 USOD10K_assets 下的 v16-lsui 目录。执行 P2 时另准备 v16-uieb。精确 run、epoch、SHA-256 及 P1 的 FX 迁移项见 [部署前资产核查](PREDEPLOY_AUDIT.md)。FX 可部署在任意指定目录，P1 支持用 --fx-root 覆盖配置。

在 AutoDL 独立取得作者固定版本的源代码。

```bash
assets=/root/autodl-tmp/pro/publicdata/USOD10K_assets
git clone --filter=blob:none --no-checkout https://github.com/LinHong-HIT/USOD10K.git "$assets/USOD10K-source"
git -C "$assets/USOD10K-source" sparse-checkout set --no-cone '/TC_USOD/Models/*.py'
git -C "$assets/USOD10K-source" checkout --detach fcdd53926054c79c20a64d1c69621316c5562929
```

通过 Git tree 可以核对 Test 名单，即使作者预测图没有检出到磁盘。完整作者模型通过 TCUSOD.pth 加载。

## 建立环境与路径配置

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pax
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN-usod
python -m venv --system-site-packages .venv
export PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pip install -r configs/downstream_usod/requirements.txt
"$PYTHON" -m pip check
cp -n configs/downstream_usod/paths.example.yaml configs/downstream_usod/paths.local.yaml
```

编辑 paths.local.yaml。Linux 目标 GPU 为 cuda:0。下表给出本次建议位置，配置也可使用其他实测资产路径。相对路径以 YAML 文件所在目录为基准。

| YAML 项 | AutoDL 位置 |
| --- | --- |
| output_root | /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN/experiments/downstream_usod/tc_usod_20261119_v1 |
| usod10k.test_root | /root/autodl-tmp/pro/publicdata/USOD10K/USOD10K-TE |
| tc_usod.repository | /root/autodl-tmp/pro/publicdata/USOD10K_assets/USOD10K-source |
| tc_usod.checkpoint | /root/autodl-tmp/pro/publicdata/USOD10K_assets/TCUSOD.pth |
| v16.lsui.config | /root/autodl-tmp/pro/publicdata/USOD10K_assets/v16-lsui/config_resolved.yaml |
| v16.lsui.checkpoint | /root/autodl-tmp/pro/publicdata/USOD10K_assets/v16-lsui/best/best_psnr.pt |
| v16.uieb.config、v16.uieb.checkpoint | P2 可选，按 v16-uieb 中同样的文件结构 |
| fx_root、fx | 配置实际部署的 FX 根和各方法的 checkpoint、source、python、sha256 |

## 执行与恢复

在 feature 的 AutoDL 检出目录执行，填入 Mac 发布时的完整 SHA。check_autodl_assets.sh 审核 CUDA 环境、作者代码、1,026 个 RGB/Depth/GT 三元组、权重哈希及两模型严格加载。

```bash
release=填写已发布的完整Git提交SHA
paths=configs/downstream_usod/paths.local.yaml
bash scripts/downstream_usod/check_autodl_assets.sh --commit "$release" --paths "$paths"
bash scripts/downstream_usod/run_p0.sh --commit "$release" --paths "$paths"
```

P0 先用真实资产在最大尺寸样本和按 ID 等距选择的样本上完成 CUDA smoke，保存验收记录后运行全量。使用相同运行配置和输出目录重跑同一条命令即可续跑。进度以每个样本的 PNG 与 SHA-256 收据为准，完整输出将被校验后复用。样本级错误位于 logs。需要变更实验资产、batch 或分块协议时，为新的 P0 指定新输出目录。

```bash
bash scripts/downstream_usod/run_p1.sh --commit "$release" --paths "$paths"
# 或显式限定已迁移的 FX 组
bash scripts/downstream_usod/run_p1.sh --commit "$release" --paths "$paths" --fx-methods waternet marine_net

# P2
bash scripts/downstream_usod/check_autodl_assets.sh --commit "$release" --paths "$paths" --phase p2
"$PYTHON" -B -m src.downstream_usod.cli smoke --commit "$release" --paths "$paths" --phase p2
"$PYTHON" -B -m src.downstream_usod.cli pipeline --commit "$release" --paths "$paths" --phase p2
```

单独阶段通过 CLI 的 prepare、enhance、saliency、evaluate、analyze、visualize 子命令访问。用各子命令的 --help 查看参数。P0、P1、P2 共用固定 Test；后续两阶段需要已完成的 P0。

### 补齐 P1 到九个 FX 方法

将新配置的 `output_root` 指向独立目录，`--reference-run` 指向已有 P0 和五个 FX 方法均完整的正式实验。使用原实验的 TC-USOD 环境和资产路径。新 FX 配置包含 PUGAN 的三个 Par 权重、UVZ 的增强权重和内部深度权重、HDAMS-Net 的正式权重，以及 IMIF 在 LSUI 验证通过的固定算法状态。每个附加文件在 `fx.<method>.auxiliary` 中配置 `path` 和 `sha256`，键名见路径模板。HDAMS-Net 的 FX 目录需保留原源码及 `validation/source_provenance.json`。

```bash
python -B -m src.downstream_usod.cli extend-p1 \
  --paths /path/to/paths-p1-fx9.yaml --commit "$commit" \
  --reference-run /path/to/completed/five-method-run
```

命令核验原实验的逐图输入和预测收据，补跑 PUGAN、UVZ、IMIF、HDAMS-Net。TC-USOD 始终读取同一份 Test Depth。汇总、配对统计和可视化写入新目录，复用来源记录在 `audit/p1_reference.json`。

### SWLGM 矩形输入

SWLGM 源码 `dfa27b81e6d3398a88cdf8e53871dcf96096588b` 的 MSFFN 将张量还原为 `B,C,W,H`，导致 640×480 输入无法完成残差相加。`scripts/downstream_usod/patches/swlgm_msffn_hw.patch` 仅将形状还原为 `B,C,H,W`。Mac CPU 验证了两种方形尺寸的逐元素一致性，以及横向和纵向矩形输入的输出尺寸和有限性。训练权重、32 像素对齐及 PNG8 导出沿用原 FX 路径。

在独立的 FX 部署副本中核验并应用补丁，随后给 P1 命令加上 `--fx-root` 指向该副本。文件位于 SWLGM 论文目录的 `SWLGM-main/basicsr/models/archs/restormer_arch.py`。以下两个摘要可用于核对补丁前后的状态。

| 源文件 | SHA-256 |
| --- | --- |
| 作者版本 | d7abc06924a3d5724ba95829ac38079e0ef58b8b20c504e4325a792c7b9730bd |
| 应用尺寸兼容补丁后 | 077f28b475880bffe141796a748b0c2ec17ad9b2746571dd775f92859a13e209 |

P1 的逐样本收据会记录 FX 源代码摘要。重试时将旧版 SWLGM 的烟测目录和汇总保存为历史记录，以便在同一 P0 发布版本下重新验证该方法。

## 结果结构与回传

```text
tc_usod_20261119_v1/
  audit/                     资产、环境、正式 run、smoke 与逐样本收据
  manifests/test.json        固定 Test、尺寸及输入哈希
  smoke/                     真实资产的少量烟测输出
  enhanced/<method>/          原尺寸 RGB PNG8
  predictions/<method>/       原尺寸灰度 PNG8
  metrics/<method>/           per_image.csv、threshold_curves.csv、per_image_curves.npz、overall.json
  metrics/<phase>/            overall.csv、paired_per_image.csv、paired_bootstrap.csv/.json
  visualizations/<phase>/     选样清单、Input/GT/各条件预测图
  logs/                       样本错误与阶段日志
  reports/<phase>_summary.*   阶段验收和汇总，P1 同时列出 FX 状态
```

正式阶段通过的依据是 reports 中的完整状态、全量输入与输出计数、各组逐样本收据及 paired bootstrap 文件。P1 有不可用方法时报告实际覆盖；已纳入方法仍逐一要求完整 1,026 图。P2 汇总应同时保留 Raw、LSUI V16 和 UIEB V16 三个条件。

```bash
out=/root/autodl-tmp/pro/UIE4_workspace/UIE4_VN/experiments/downstream_usod/tc_usod_20261119_v1
bash scripts/downstream_usod/export_results.sh --output-root "$out" \
  --archive "$out/reports/usod_p0_summary.tar.gz" --phase p0 --include-figures
```

从 Mac worktree 下载指定汇总包。包内有 SHA256SUMS，默认归档轻量表格和报告，--include-figures 另加入选样表及一张总图。

```bash
bash scripts/downstream_usod/download_results.sh \
  /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN/experiments/downstream_usod/tc_usod_20261119_v1/reports/usod_p0_summary.tar.gz \
  ./usod_p0_summary.tar.gz
```
