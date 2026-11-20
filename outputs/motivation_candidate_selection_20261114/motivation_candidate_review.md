# ICAR-UIE 首页动机图候选复核

## 交付与证据

LSUI 和 UIEB T90 各生成 10 张候选，每套为 A 组 4 张、B 组 4 张、C 组 2 张。四列依次为 Input、Global RGB Shift Oracle、Remaining Correction Magnitude 和 Reference。总览 PNG 为 2842 × 6425 像素、300 dpi。PDF 保留可选取的文字、矢量刻度和分隔线，四个面板保留各自的 256 × 256 RGB 位图。Reference 用于人工核查。单栏试排只放 Input、Oracle 和幅值图，每行宽 236.25 pt（83.34 mm），以 96 dpi 查看。

[LSUI 总览 PNG](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/LSUI_10_candidates.png) · [LSUI PDF](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/LSUI_10_candidates.pdf) · [UIEB T90 总览 PNG](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/UIEB_10_candidates.png) · [UIEB T90 PDF](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/UIEB_10_candidates.pdf) · [三面板单栏试排](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/single_column_proof.png)

[20 张逐图统计与单图路径](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/motivation_candidate_ranking.csv) · [选样和人工复核记录](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/candidate_selection.json) · [运行参数与验证结果](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/reproduction_parameters.json)

### 发布协议和字段

| 显示或输出字段 | 与发布实验的对应关系 |
| --- | --- |
| RGB Shift R/G/B、`||μ||₂` | `mu_R/G/B` 是各通道的逐图平均 `Y − I`。L2 是这三个已保存通道值的欧氏范数。 |
| Raw Error、Global Oracle Error | `e_raw = mean(Δ²)`，`e_global = mean(r²)`，分别在整张图的所有像素和通道上求均值。这里 `r = Δ − μ`。 |
| `η` | `eta_spatial = e_global / (e_raw + 10⁻¹²)`。平方误差的实际下降比例为 `1 − e_global / e_raw`。 |
| `V16` | `patch_variance_16`。在 256 × 256 图像左上角起按 16 × 16 **像素**作不相交的等面积块。先求每个块的三通道平均 `Δ`，再求各通道的块间总体方差（`ddof=0`），最后对三个通道取平均。该网格恰好含 16 × 16 块，无补边或裁边。减去常量 `μ` 后，这个量也等于块平均剩余 `r` 的平方能量。 |
| `V16 / E_global` | `patch_spatial_retention_16`。分母为逐图未裁剪剩余平方误差。 |
| Oracle clip | 主列为 `oracle_pixel_clipped_fraction`，即 `I + μ` 至少一通道越界的空间像素比例。CSV 另列 scalar fraction，按所有 RGB 分量计数。 |
| Magnitude above display max | `display_remaining_clipped_fraction`，即未裁剪剩余的 RGB L2 模长超过 1 的像素比例。 |

严格 RGB 解码，Input 和 GT 均用 Pillow 双线性插值变为 256 × 256，再用 float32 除以 255。残差、均值和误差统计用 float64。Oracle 图像显示时裁剪至 [0,1]。幅值始终由未裁剪的 `r` 计算，20 张共享线性 `magma` [0,1]，使用与发布实验同版本的 Matplotlib 3.11.2 查色表。

[LSUI 协议](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/motivation_correction_demand/run_20261106_005616_delivery/lsui/metadata/protocol.json) · [UIEB T90 协议](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/motivation_correction_demand/run_20261106_005616_delivery/uieb/metadata/protocol.json) · [原分析实作](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/tools/analyze_motivation_correction_demand.py:159) · [原发布独立验证实作](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/tools/verify_motivation_correction_demand.py:46)

### 数值和源图验证

- 发布版测试清单、逐图 CSV 和源文件哈希清单逐行对齐。LSUI 428 对、UIEB T90 90 对，ID 唯一，无失败记录。UIEB 的 82 对 `Train/` 路径和 8 对 `Val/` 路径都由发布版 `test.tsv` 定义为 T90 配对。
- 本地项目使用工作区副本。LSUI 发布根 `/home/paxton/pro/publicdata/LSUI19_dup_train` 不在本机，来源为哈希匹配的本地 `publicdata/LSUI`（784 个文件）和 `publicdata/LSUI19`（72 个文件）。UIEB T90 的 180 个文件均来自本地 `publicdata/UIEB19`。518 对、1,036 个文件全部逐个匹配归档 SHA-256，源图缺失为 0。逐图表保留本地实读路径和发布路径。
- 对全体 518 对重新核对 11 个直接用于筛选或显示的字段。LSUI 最大绝对差为 `7.216449660063518e-15`，UIEB T90 为 `1.887379141862766e-15`，均低于发布独立验证器的严格绝对容差 `2e-10`。本地 Pillow 13.0.0 和 NumPy 2.3.5 与原环境版本不同。六张历史样本的七个已归档数组与本地重建逐元素相等，复原的四面板像素亦与各自发布 SVG 对齐取样后的像素逐点相等。
- `320`、`213_img_`、`605_img_` 使用保存的 `numeric_maps.npz`，其余 17 张由哈希匹配的原始配对复建。两张总览 PDF 中的 80 个嵌入面板都逐字节核对，均等于已验证的可视化数组。113 个发布资产在生成前后均匹配发布清单。
- 已核查候选的 Input 与 Reference，未见明显配对错误。视觉复核基于现有 256 × 256 图像和单栏试排，原分辨率亚像素配准与参考图制作过程未验证。UIEB `3001`、`6062`、`6820` 的个别角落有原图标记，幅值图中可见其局部痕迹。三张仍可用于筛选，Top 3 采用其他样本。
- 20 张中有 5 张 Oracle 存在少量显示裁剪。LSUI `167` 0.081%、`2414` 0.031%、`4092` 0.040%、`3589` 1.118% 和 UIEB `6820` 0.143%。其余 15 张为 0%，所选 20 张中幅值超出 1 的像素比例均为 0%。未发现本次复算无法重现的所查字段或显示协议冲突。

[LSUI 逐图统计](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/motivation_correction_demand/run_20261106_005616_delivery/lsui/statistics/per_image_metrics.csv) · [UIEB T90 逐图统计](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/motivation_correction_demand/run_20261106_005616_delivery/uieb/statistics/per_image_metrics.csv) · [发布资产清单](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/motivation_correction_demand/run_20261106_005616_delivery/artifact_inventory.json)

## 选样规则

分位在各自完整测试集内计算，LSUI 分母 428，UIEB T90 分母 90。分位为 `100 × (低于该值的图数 + 与该值相等的图数 / 2) / N`。区间左闭右开。通用门槛为 Oracle 像素裁剪 ≤5%、RGB 分量裁剪 ≤2%、幅值超色标比例为 0、`E_global` ≥ P25。这些门槛用于本次候选图筛选。

| 组别 | 细分区间与共同条件 |
| --- | --- |
| A 全局变化 | 按 `||μ||₂` 的 [P45,P60)、[P60,P72)、[P72,P85)、[P85,P97) 各留一张。限 `η` 在 10%–65%、`V16` ≥ P25。优先检查低裁剪、偏移区间中央、方差及保留率较高、或误差下降适中的池内图像。 |
| B 空间剩余 | 按 `V16` 的 [P55,P68)、[P68,P78)、[P78,P90)、[P90,P98) 各留一张。限保留率 ≥ P55、`||μ||₂` 在 P25–P95、`η` 在 12%–80%。复核保留率较高、块方差区间中央、或偏移较大的图像。LSUI B2 初看有带显著水印的近似场景，扩查该档全部 21 张。 |
| C 综合平衡 | C1 取 `||μ||₂` [P25,P55)、`V16` [P25,P65)，C2 取 [P55,P80)、[P37,P82)。两档均要求 `η` 在 [P25,P80)、保留率 ≥ P25、`E_global` ≤ P86。优先看居中组合，同时保留可见色变和清楚内容。 |

在候选池内按不同侧重点查看图像，然后根据配对观感、主体辨识、固定色标下的连续分区和单栏表现定档。每档实际看过的 ID 与最终名额已记录，历史发布的六张另做全覆盖复核。初筛区间、视觉判断和 Top 3 都属于为论文插图设定的有目的筛选，文稿中宜称 *illustrative examples*。表中顺序是组别与强度区间。三张推荐图是人工复核后的排序，保留指标间的区别。

| 数据集 | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSUI 合格池 | 32 | 41 | 45 | 39 | 23 | 21 | 37 | 24 | 22 | 41 |
| UIEB T90 合格池 | 12 | 9 | 7 | 8 | 3 | 5 | 6 | 4 | 5 | 11 |

## 逐图入选理由

### LSUI

| 档 | ID | 视觉复核 |
| --- | --- | --- |
| A1 | 350 | 中档偏移就能减轻绿色偏色，中央岩石及左侧暗区形成成片剩余。 |
| A2 | 1363 | 礁面从青色出现较明显的暖色成分，水体与珊瑚前景的剩余分区清楚。 |
| A3 | 644 | 整体绿偏变化醒目。平岩、沟槽和上方水体的剩余区在单栏仍可区分。 |
| A4 | 441 | 高档偏移明显调整绿偏，斜向岩层留下大尺度剩余。 |
| B1 | 167 | 海胆、浅色岩顶和下方阴影提供连贯结构，属于 B 组较低方差档。 |
| B2 | 2414 | 鱼体的综合色变易辨认，外缘和下腹的剩余连续。 |
| B3 | 320 | 岩沟和海胆周围的成片区缩小后仍可见，块能量保留 88.84%。 |
| B4 | 827 | 亮水体与右侧岩壁之间的剩余分区强，块能量保留 96.73%，处在数据集最高端。 |
| C1 | 4092 | 中低档偏移和中档方差，鱼群与开阔水体可区分，剩余幅值比强例温和。 |
| C2 | 3589 | 偏移、方差和保留率都处于中段，偏色变化可见，枝状珊瑚余差较暗。 |

### UIEB T90

| 档 | ID | 视觉复核 |
| --- | --- | --- |
| A1 | `605_img_` | 中档偏移即可明显减绿，珊瑚前景与上方亮水体的连片剩余可辨。 |
| A2 | `208_img_` | 器物与沙底的青偏减轻，块方差处在中段，顶部和器物之间仍有成片剩余。 |
| A3 | `213_img_` | 水体由棕灰向蓝色转变，礁区和水体仍有可见剩余。 |
| A4 | 1660 | 浅底与深色鱼体面积较大，大幅偏移和连片剩余都便于缩小观察。 |
| B1 | 2787 | 鱼腹和背景的剩余对照清楚，在 B 组较低档方差下已有连续区域。 |
| B2 | 3001 | 鱼体、礁面和上方水体有清楚分区，含轻微角落原图标记。 |
| B3 | 6062 | 潜水员与橙色弧面对应大面积分区，保留率 89.03%，含轻微角落原图标记。 |
| B4 | `602_img_` | 高块方差，浅底与深礁沿对角线呈大片对照，同时可见减绿。 |
| C1 | 9896 | 偏移幅值及剩余误差占比接近数据集中心，前景珊瑚和水体的余差较轻且连贯。 |
| C2 | 6820 | 双潜水员与水体的对照清楚，偏移与方差居中，含轻微角落原图标记。 |

突出全局校正变化时，优先对比 LSUI `644`、`441` 与 UIEB T90 `1660`、`208_img_`。突出空间组织时，优先对比 LSUI `320`、`827` 与 UIEB T90 `2787`、`6062`、`602_img_`。其中 `827` 可作为高保留率强例，`2787` 可作为块方差非极端且主体仍清楚的例子。

### 历史样本去留

| 样本 | 结果与理由 |
| --- | --- |
| LSUI 320 | 入选 B3。高保留率（P92.87）、无裁剪，空间剩余清楚。 |
| LSUI 3809 | 未入选。Oracle 像素裁剪 8.91%，超出 5% 筛选门槛，固定色标下剩余也较暗。 |
| LSUI 3316 | 未入选。块方差约 P19.74、保留率约 P14.14，剩余主要落在小主体，缩小后较弱。 |
| UIEB T90 `212_img_` | 未入选。`V16` 位于 P50.56，`||μ||₂` 在 P7.22，单栏下颜色变化弱。 |
| UIEB T90 `213_img_` | 入选 A3。色变明显且无裁剪，保留率为 P38.33。 |
| UIEB T90 `605_img_` | 入选 A1。中档偏移下的减绿清楚，连片幅值仍可辨。 |

[LSUI 历史选样记录](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/motivation_correction_demand/run_20261106_005616_delivery/lsui/metadata/selected_samples.json) · [UIEB T90 历史选样记录](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/motivation_correction_demand/run_20261106_005616_delivery/uieb/metadata/selected_samples.json)

## 首页备选 Top 3

| 顺序 | 数据集与 ID | `||μ||₂` 分位 | `η` 分位 | `V16` 分位 | 保留率分位 | `E_global` 分位 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | UIEB T90 [2787](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/individual_samples/uieb/B1_2787.png) | P70.56 | P36.11 | P55.00 | P70.56 | P40.56 |
| 2 | LSUI [644](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/individual_samples/lsui/A3_644.png) | P84.46 | P62.73 | P95.44 | P87.50 | P94.98 |
| 3 | UIEB T90 [605_img_](/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/individual_samples/uieb/A1_605_img_.png) | P50.56 | P80.56 | P87.22 | P62.78 | P89.44 |

1. **2787** 在单栏下可辨认鱼体、鱼腹的连续高幅值区及背景。最优常量偏移将平方误差减少约 67.46%，仍有清楚的空间结构，`V16` 处在 P55。
2. **644** 减绿明显，平岩和沟槽的空间结构在固定色标下最易辨认。平方误差减少约 64.91%。`V16` 位于 P95.44，适合作为较强例。
3. **605_img_** 用中档偏移即可看清颜色变化，上部亮水体和珊瑚保留大片剩余。平方误差减少约 36.98%，适合表现偏移有效但余差仍显著的情形。

三张 Oracle 的像素裁剪均为 0%。图例能说明在该 GT-dependent 逐图常量 RGB 加性偏移下仍存在空间组织的剩余校正需求。此结论的比较对象限定为单个逐图常量加性偏移。

## 复现

`candidate_selection.json` 固定分位区间、阈值、目视复核的 ID、每档入选理由及 Top 3。`generate_candidates.py` 对照全测试集核验阈值、哈希和逐图数值，再按人工复核清单重建资产。`reproduction_parameters.json` 保存实际运行命令、各输入版本/哈希、候选偏差及色标参数。复现时将最后的路径指向一个新目录。

```sh
"/Users/paxton/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3" -B \
  "/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_20261114/generate_candidates.py" \
  --project "/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN" \
  --publicdata "/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/publicdata" \
  --matplotlib-colormap "/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/analysis/model_complexity_20261106/.deps/matplotlib/_cm_listed.py" \
  --font-dir "/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/analysis/model_complexity_20261106/.deps/matplotlib/mpl-data/fonts/ttf" \
  --pdftoppm "/Users/paxton/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm" \
  --output "/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/outputs/motivation_candidate_selection_rerun"
```

