# V16 的 SIFT 特征匹配评估（Windows WSL / RTX 3060 12GB）

本工具使用现有 V16 权重或已保存的增强结果，计算 SIFT 特征匹配指标并生成彩色连线图。无需重新训练。所有数据和权重继续使用你已搬迁到 WSL 的文件，通过参数覆盖原服务器路径。

入口：`tools/evaluate_sift.py`。这是额外评估，不会改变 V16 的网络、训练脚本、数据划分或原有 PSNR/SSIM 结果。

## 1. 同步代码并检查环境

将以下新增文件按原目录结构复制到 WSL 的 **UIE4_VN 项目根目录**：

```text
tools/evaluate_sift.py
tools/sift_eval/__init__.py
tools/sift_eval/matching.py
requirements-sift.txt
configs/sift/pairs.example.csv
README_SIFT.md
tests/test_sift_evaluation.py       # 可选：软件自检
```

如果使用交付的 `UIE4_VN_SIFT_WSL.zip`，它只包含上述新增文件。在 WSL 中进入现有项目后解压即可。例如 ZIP 在 Windows 下载目录时：

```bash
cd /home/你的WSL用户名/projects/UIE4_VN
unzip /mnt/c/Users/你的Windows用户名/Downloads/UIE4_VN_SIFT_WSL.zip
```

上面的路径仅是示例，必须替换成你的真实路径。然后激活你原来运行 V16 的 Python/conda 环境，在 **WSL 终端** 中检查：

```bash
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -c "import cv2; print('OpenCV:', cv2.__version__, 'SIFT:', hasattr(cv2, 'SIFT_create'))"
python tools/evaluate_sift.py --help
```

若 `cv2` 不存在，安装额外依赖：

```bash
python -m pip install -r requirements-sift.txt
```

如果已有可用的 `cv2.SIFT_create`，继续使用现有 OpenCV 即可，不必再安装另一种 OpenCV 发行包。现有 V16 环境提供 PyTorch、NumPy、Pillow、PyYAML；本工具不需要重新安装 CUDA、训练模型或下载预训练特征网络。

## 2. 推荐先跑：现有 LSUI / UIEB 的 GT 对照

这一模式分别匹配：

```text
原始输入       ↔ 固定的 GT
V16 最终输出   ↔ 同一个 GT
```

两次匹配的参考图完全相同。它测量的是“与对齐参考图的特征一致性”，不能当作跨视角匹配、SLAM 或定位精度。

设置你在 WSL 中的真实路径，只需要替换下面两项：

```bash
V16_RUN="/home/你的WSL用户名/projects/UIE4_VN/experiments/你的V16实验目录"
UIE_DATA="/home/你的WSL用户名/datasets/LSUI19"
```

`V16_RUN` 目录应包含：

```text
config_resolved.yaml
best/best_psnr.pt
split_snapshot/test.tsv
```

`UIE_DATA` 是包含 `Train`、`Val` 等子目录的数据根目录，必须能与冻结 TSV 中的相对路径拼接。UIEB 同理，指定对应 UIEB 权重和 UIEB 数据根目录。不要把不同数据集训练的实验路径混在一起。

**先用 10 张检查整个流程：**

```bash
python tools/evaluate_sift.py gt \
  --run-dir "$V16_RUN" \
  --data-root "$UIE_DATA" \
  --device cuda:0 \
  --limit 10 \
  --output-dir "$V16_RUN/result/sift_gt_smoke"
```

**确认输出后，跑完整冻结测试集：**

```bash
python tools/evaluate_sift.py gt \
  --run-dir "$V16_RUN" \
  --data-root "$UIE_DATA" \
  --device cuda:0
```

默认读取 `best_psnr`（或运行配置中记录的默认选择），严格检查 checkpoint 的 V16 版本、模型配置和权重键。使用 `--checkpoint best_ssim` 可明确选择其他已有权重。请固定用于论文报告的 checkpoint，不要根据测试集 SIFT 分数挑选权重。

默认使用运行目录中的 `split_snapshot/test.tsv`，并检查配置记录的测试集数量。若搬迁时遗漏快照，应优先恢复原快照；也可以明确指定 `--manifest split/lsui19/test.tsv` 或对应 UIEB 的固定清单。只有显式给出参数才会使用这个替代清单，结果会记录其路径和 SHA256。

## 3. 如果已经有 V16 增强结果，可直接评估

官方测试产生的 `result/test_all_enhanced` 可以直接复用：

```bash
python tools/evaluate_sift.py gt \
  --run-dir "$V16_RUN" \
  --data-root "$UIE_DATA" \
  --prediction-dir "$V16_RUN/result/test_all_enhanced" \
  --output-dir "$V16_RUN/result/sift_gt_saved"
```

这条命令不运行 V16，也不需要 GPU。SIFT 本身在 CPU 上运行。输出图像名称按照官方 V16 测试规则对应到清单，尺寸必须与当前评估设置一致；脚本不会把已有预测偷偷缩放成所需尺寸。提供的预测是否来自所选实验需要由使用者保证，报告会明确标记“未加载 checkpoint”。

## 4. 分辨率与 12GB 显存

默认 `--resolution config` 沿用 `config_resolved.yaml` 的 `evaluation` 设置；现有 V16 标准配置为 256×256。每次推理只有一张图，不创建训练梯度，也不使用 DataLoader 多进程，适合作为 3060 12GB 的起步设置。

若要观察更大分辨率下的局部特征，可以单独运行一个实验，保持比例、长边最多 512 像素：

```bash
python tools/evaluate_sift.py gt \
  --run-dir "$V16_RUN" \
  --data-root "$UIE_DATA" \
  --device cuda:0 \
  --resolution native --max-side 512 \
  --output-dir "$V16_RUN/result/sift_gt_max512"
```

显存允许再尝试 768。`--resolution native --max-side 0` 使用原始尺寸，显存需求取决于具体图像，12GB 不能保证容纳所有原图。可以显式添加 `--amp` 启用 CUDA 半精度；它可能改变少量输出像素和匹配结果，因此报告会记录精度模式，比较方法时必须保持一致。默认是 FP32。

脚本不会自动降分辨率或切块。V16 的 UICF 和 Color Query 含有全局信息，切块会改变推理含义；本工具直接调用现有模型的完整 forward 和内置 pad/crop。遇到显存不足，会提示你明确调整设置后重跑。

比较多个方法时，固定输入、图像尺寸、量化方式、SIFT 参数、几何模型和阈值。`--threshold-px` 的单位是**实际参与匹配的评估图像像素**，不同分辨率的分数应分开报告。

## 5. 最接近示例图：真实重叠场景图像对

准备同一场景、有重叠区域的两张不同照片，或视频相邻帧。LSUI/UIEB 的“输入与 GT 成对”不等同于这种跨视角场景对；不能随机取两张无关的测试图代替。

复制并编辑 `configs/sift/pairs.example.csv`。最少需要三个 CSV 列：

```csv
pair_id,image_a,image_b
scene01,raw/frame001.jpg,raw/frame002.jpg
scene02,raw/frame010.jpg,raw/frame011.jpg
```

图像路径相对于 `--data-root`，也可以明确使用 WSL 绝对路径。`pair_id` 必须唯一，重复或反向重复的图像对会被拒绝。不会自动挑选匹配变好的图像对。

例如：

```bash
python tools/evaluate_sift.py pairs \
  --run-dir "$V16_RUN" \
  --data-root /home/你的WSL用户名/datasets/feature_pairs \
  --pairs configs/sift/pairs.example.csv \
  --device cuda:0 \
  --resolution native --max-side 512 \
  --geometry homography \
  --output-dir "$V16_RUN/result/sift_scene_pairs"
```

每对图像都分别执行：

```text
原始 A        ↔ 原始 B
V16 增强 A    ↔ V16 增强 B
```

`homography` 适合近似平面或可用单应性描述的视角变化。明显存在三维视差的场景，可明确使用 `--geometry fundamental`，它使用基础矩阵的 RANSAC，至少需要 8 个候选匹配。几何模型要根据场景事先选定，不能根据哪个模型分数更高来选择。

若这些照片已经被增强，在 CSV 中增加 `enhanced_a,enhanced_b` 两列，其路径相对于 `--prediction-dir`，然后给上面的命令添加 `--prediction-dir /你的增强结果目录`。这时无需重新推理，两个增强图也必须与对应原图在评估分辨率下的尺寸一致。

## 6. 指标与结果文件

默认完整 GT 结果在 `$V16_RUN/result/sift_gt/`：

```text
per_pair_metrics.csv        # 每对图像两行：raw / v16
sift_evaluation.json        # 汇总、协议、参数、软件版本和来源
sources.json               # 输入/参考/已有预测的哈希与原始/评估尺寸
matches/                   # 每对图像的匹配坐标、描述子距离、内点标记、几何矩阵
figures/overview.png       # 最多六组对比图；每组上排原始、下排 V16
figures/*_comparison.png   # 单组上下对比
figures/*_raw.png           # 单独的原始匹配图
figures/*_v16.png           # 单独的 V16 匹配图
enhanced/                  # 由本次推理生成的 RGB PNG8；复用预测时没有此目录
```

CSV 主要字段：

| 字段 | 含义 |
|---|---|
| `keypoints_a/b` | 两张图的 SIFT 特征数量 |
| `ratio_matches_a_to_b` | A→B 的 Lowe ratio test 通过数量 |
| `tentative_matches` | 默认再经双向 ratio 互检后的候选数量 |
| `verified_matches` | 通过指定几何检查的候选数量 |
| `inlier_ratio` | verified / tentative；无候选时记 0 |
| `geometry_status` | 正常、无特征、数量不足或估计失败 |

GT 模式利用已知对齐关系，检查两端坐标距离是否不超过默认 3 像素，不拟合一个新单应矩阵去掩盖模型产生的位移。真实场景模式分别对 raw、V16 估计几何模型，报告的是各自的几何一致性，不是真实标注的匹配准确率。

默认 SIFT 参数为 contrast threshold=0.04、edge threshold=10、sigma=1.6、nfeatures=0（不封顶），BF L2 的 k=2 匹配，ratio=0.75，并要求两向 ratio 匹配互相对应。RANSAC 固定 confidence=0.999、maxIters=5000、seed=3407；OpenCV 默认单线程并关闭 OpenCL。汇总保存实际 OpenCV/NumPy/Pillow 版本；跨版本或硬件仍可能出现差异。

汇总同时保存候选/验证匹配的均值、总数、内点率的宏平均和微平均、增加/持平/下降的图像对数量。所有无特征和几何失败样本均保留，不能只报告成功的样本。原始验证匹配总数为 0 时，相对增益记为 JSON `null`，避免用无意义的无穷大增益。

图上的彩色线只显示几何验证后的匹配，标题同时标注总数和实际画出的数量。默认 `--max-draw 0`，画出全部通过检查的匹配，不限制连线数量；如果需要限制显示数量，可以显式使用如 `--max-draw 200`，显示上限不改变统计。可视化样本在评价前按固定随机种子选择，不按收益筛选；`--num-visualizations` 控制保存数量。图像缩放仅用于展示，发生在指标计算之后。

带 `--limit` 的结果会标记 `partial: true`，只用于预跑。完整数据均已处理时才得到全清单结果。已有输出默认拒绝覆盖；明确添加 `--overwrite` 可替换此工具生成的 SIFT 目录，计算失败时保留上一次完整结果。

这种实验适合作为图像增强的补充证据，应与图像质量、失真和下游任务结果一起解释。匹配数量更多不必然更好，噪声、伪纹理和重复纹理同样可能产生候选匹配。截图没有提供原论文完整参数，本工具定义了可记录、可重复的评估协议，不声称逐参数复现截图论文。

实现参考：[OpenCV SIFT/特征匹配](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)、[OpenCV 几何验证示例](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)。

## 7. 软件自检

```bash
python -m pytest -q tests/test_sift_evaluation.py
```

测试使用已知几何变换的合成纹理、空白图、少量候选及退化点集，检查匹配与几何验证、失败样本的统计、已有输出复用、事务式输出保护，以及小尺寸真实 V16 工厂模型的 checkpoint 加载与推理。测试中的随机权重和合成图分数只代表软件检查，不能当作你的 V16 实验结果。
