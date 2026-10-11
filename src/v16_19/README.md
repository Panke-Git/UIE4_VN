# V16-19：独立 V16 模型与 Train/Val 两集合实验

V16-19 保留 V16 的 `PlainUNetColorQuery(UICF(I))` 模型结构，直接使用当前 LSUI19/UIEB19 的物理 Train/Val 划分。训练完成后，重新加载指定的最佳 checkpoint，导出完整 Val 结果。版本标识为 `v16_19`，展示名称为 `V16-19`。

| 数据集 | 梯度训练 | 验证及 checkpoint 选择 | 最终 Val 导出 |
| --- | ---: | ---: | ---: |
| LSUI19 | Train：3,851 对 | Val：428 对 | 同一组 428 对 |
| UIEB19 | Train：801 对 | Val：89 对 | 同一组 89 对 |

本版本没有独立测试集；没有 test 入口、test.tsv、T90 划分或 test 结果。Val 同时参与模型选择，最终指标应称为验证集成绩。

## 隔离范围

- `src/v16_19/` 包含自己的数据加载、训练、验证、运行管理、指标和模型实现；运行时不导入 `src/v16`、`src/v4` 或 `src/shared`。
- V4 U-Net、Color-Query、UICF 和 CIEDE2000 的必要实现固定在新模块内。`snapshot_provenance.json` 记录原始文件来源和哈希。
- 新配置仅位于 `configs/v16_19/`；清单位于 `split/v16_19/`。
- 输出强制位于 `experiments/v16_19/<dataset>/`，每次新训练创建不同的时间戳目录。
- 旧 V16 的代码、配置、清单、checkpoint 和结果不修改。新版本拒绝旧版本 checkpoint 续训或验证。

## 环境与工作目录

以下命令从 `UIE4_VN` 项目根目录执行。复用现有 PyTorch 环境；必要依赖是 PyTorch、NumPy、Pillow、PyYAML、`scikit-image>=0.25,<0.26`。测试另需 pytest。图片导出及曲线绘制使用 Pillow，不额外依赖 matplotlib。

两个配置默认的数据根目录分别为 `/root/autodl-tmp/pro/publicdata/LSUI19` 和 `/root/autodl-tmp/pro/publicdata/UIEB19`。在这台服务器上执行下方 preflight 和训练命令即可，无需额外传入 `--data-root`。若数据位于其他目录，使用 `--data-root /absolute/path` 覆盖配置，不需要修改清单；相对路径始终按项目根目录解析。

## 数据准备与冻结

当前本地真实数据已生成两套完整的冻结清单。部署到服务器时，请上传 `split/v16_19/` 和对应的原始数据，再执行验证。

```bash
python -m src.v16_19.prepare --dataset lsui19 --data-root /root/autodl-tmp/pro/publicdata/LSUI19 --verify-only
python -m src.v16_19.prepare --dataset uieb19 --data-root /root/autodl-tmp/pro/publicdata/UIEB19 --verify-only
```

如果用于首次准备相同协议的新 checkout，去掉 `--verify-only`。不存在输出目录时，工具根据实际物理 Train/Val 文件生成清单；已有输出目录时只检查，不重新划分或覆盖。CLI 输出目录受限于 `split/v16_19/`，因此不会覆盖旧清单。已有清单和数据不匹配时应核对数据，而不是删除清单来掩盖变化。

每套协议包含：

- `train.tsv`、`validation.tsv`：无表头，三列为 sample_id、相对 input 路径、相对 GT 路径，UTF-8、按文件名排序。
- `file_index.json`：全部 input/GT 的文件 SHA256、RGB 像素哈希、尺寸和真实编码。
- `data_quality.json`：精确重复输入、跨集合重复、相同输入的 GT 一致性、扩展名与真实编码差异。
- `protocol.json`：数据集、两集合数量、上述文件哈希和整体 fingerprint。

图像数量、配对、可读性、RGB 模式和尺寸均在准备时检查。训练前、续训前及最终评估前后检查原文件字节、完整成员及清单 fingerprint。缓存 sidecar `.DS_Store`、`._*` 被忽略，其他额外条目拒绝。

固定物理划分保留原有内容重复：当前 LSUI19 有 36 组精确重复输入，其中 9 组跨 Train/Val；UIEB19 有 7 组，均在 Train 内，无跨 Train/Val 精确重复。此处指尺寸及 RGB 像素完全相同，不包括近似场景重复。数据不会自动去重或移动。

## 训练前检查

```bash
python -m src.v16_19.preflight --config configs/v16_19/lsui19.yaml --gpu 0
python -m src.v16_19.preflight --config configs/v16_19/uieb19.yaml --gpu 0
```

preflight 检查完整文件身份、配置、模型和一张真实 Val 图像的前向及指标计算。它不训练，也不声称完成全量 Val 评测。没有 CUDA 时使用 CPU。

## 两个数据集分别训练

```bash
python -m src.v16_19.train --config configs/v16_19/lsui19.yaml --gpu 0
python -m src.v16_19.train --config configs/v16_19/uieb19.yaml --gpu 0
```

使用原本地数据路径时的覆盖示例：

```bash
python -m src.v16_19.train \
  --config configs/v16_19/lsui19.yaml \
  --data-root ../publicdata/LSUI19 \
  --gpu 0
```

两个配置保留当前 V16 的原训练配方：种子 3520，256 patch，batch size 16，200 epochs，Charbonnier loss，AdamW（lr 0.0002、weight_decay 0），无 scheduler，CUDA AMP，现有翻转与 90° 旋转增强。Val 默认按 PIL bilinear resize 到 256×256。这仅对齐 D2AFM 的数据成员和两集合流程，不复制 D2AFM 的模型或训练配方。

支持 `--seed`、`--name`、`--gpu`、`--data-root`、`--epochs`、`--batch-size`、`--num-workers`。显存不足时可在新训练开始时降低 batch size；续训不允许改变 seed、name、batch size 或 num_workers。

训练时每个验证周期计算 loss、PSNR、SSIM、ΔE00；最后一轮一定验证。默认保存 best_psnr、best_ssim、best_loss、last 和每 10 轮的 checkpoint。`validation.checkpoint` 默认 `best_psnr`，该 selector 的保存开关必须打开。

训练结束后重新构建模型并加载选定 checkpoint，而不是直接导出最后一轮模型。全部汇总指标和图片来自同一个 checkpoint。CPU 运行时 AMP 自动关闭。

## 续训与最终导出重试

```bash
python -m src.v16_19.train --resume experiments/v16_19/lsui19/<run>/checkpoint/last.pt --gpu 0
```

显式延长训练：

```bash
python -m src.v16_19.train --resume experiments/v16_19/lsui19/<run>/checkpoint/last.pt --epochs 300 --gpu 0
```

续训使用保存的 resolved config、数据快照和最新 epoch checkpoint，恢复 optimizer、AMP scaler、Python/NumPy/Torch/CUDA RNG 及 DataLoader generator。`--config` 不与 `--resume` 混用；`--epochs` 只能维持或增加已保存预算。数据路径可迁移，但图像字节和成员必须相同。

代码哈希、协议或 checkpoint 身份不匹配时拒绝续训。旧 epoch 与已有历史不一致时拒绝追加，避免重复 epoch。已完成训练的 last checkpoint 再次 resume 只重试最终 Val 导出，不重复训练；延长 epochs 则继续训练。

## 独立重新导出 Val

```bash
python -m src.v16_19.validate --run-dir experiments/v16_19/lsui19/<run> --checkpoint best_psnr --gpu 0
python -m src.v16_19.validate --run-dir experiments/v16_19/uieb19/<run> --checkpoint best_ssim --gpu 0
```

selector 支持 best_psnr、best_ssim、best_loss、last；也可给出相对 run 的 checkpoint 文件路径或绝对路径。所有 checkpoint 都校验版本、模型、代码和数据 fingerprint。

不同 selector 输出到不同目录；同一 selector 重新导出时，先在临时目录完成检查，再替换已有结果。评估失败时保留已有成功报告。指标在 PNG8 保存之前以 float RGB 计算；图像按配置评估尺寸保存，`evaluation.resize: false` 时按原分辨率逐张评估。

## 每次实验的输出

```text
<run>/
  config_source.yaml / config_resolved.yaml / config.json
  run_info.json / status.json
  code_snapshot/                  本次代码副本
  split_snapshot/                 两份清单、文件索引、质量记录、协议
  best/                           best_psnr、best_ssim、best_loss
  checkpoint/                     last、周期 checkpoint
  log/                            train、val、final_val 日志与 CSV/JSON 历史
  result/
    validation_latest/            最近一次训练中验证的指标
    val/
      best_psnr/
        summary.json              最终完整 Val 汇总及 checkpoint SHA256
        metrics.csv               Val 每张图的 PSNR、SSIM、ΔE00
        enhanced/                 全部 Val 增强 PNG
        samples/                  固定抽样 input、enhanced、GT
        visualization_samples.json
        val_grid_10x3.png          样本不足十张时自动减小
        training_curves.png
      best_ssim/ ...               独立导出其他 selector 时生成
```

`status=completed` 只在训练和最终 Val 导出均成功后写入。最终默认报告是 `result/val/best_psnr/summary.json`；`validation_latest` 是训练中的最近一次验证，两者不要混淆。输出包含具体 epoch、checkpoint SHA256、数据 fingerprint、预处理、指标定义和 `split=validation`。

## 回归检查

```bash
python -m pytest -q tests/test_v16_19.py
```

覆盖完整默认 V16 初始化权重一致性、非零 UICF 条件下前向及反向完全一致、运行时导入隔离、重复保留与文件变更拒绝、两个数据集训练/导出、原分辨率与路径迁移、旧 checkpoint 拒绝、失败导出保留结果，以及分段续训和连续训练权重逐项相等。

本地 CPU 检查不等于 CUDA 训练或正式 200 epoch 实验。正式分数来自目标训练环境完成后的 Val 报告。

本次检查结果与真实数据 fingerprint 见 [VALIDATION.md](VALIDATION.md)。
