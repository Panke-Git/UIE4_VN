"""Objective report of conditional, registered-reference feature consistency."""


def number(value):
    return '未定义（基线为零）' if value is None else f'{value:.4f}'


def percent(value):
    return '未定义（基线为零）' if value is None else f'{value * 100:.2f}%'


def write_report(output, config, unlimited, fixed, paired, intervals, regression, strongest, figure, rows):
    scope = config['scope']
    lines = ['# SIFT 参考特征一致性 V2', '',
             f"范围：{scope['evaluated_images']}/{scope['manifest_images']} 张；" + ('**部分样本 smoke test，不能作为全测试集结论。**' if scope['partial'] else '**全部冻结测试集图片。**'), '',
             '本工具仅读取保存的 Raw、增强图和 GT，未训练或推理网络。所有指标描述 **reference-geometrically consistent correspondences**：在已配准参考图下，经双向 ratio 过滤后、坐标欧氏距离 ≤3px 的对应关系。它们没有经过独立人工匹配真值确认，不能称为 ground-truth matching accuracy 或 real-world matching accuracy，也不能代表跨视角任务性能。', '',
             '## 协议与可复现性', '',
             'Raw/GT：Pillow RGB，BILINEAR 整图缩放到 256×256；增强图必须已是 256×256 RGB uint8，不进行二次缩放或几何校正。灰度为 OpenCV RGB2GRAY，无直方图均衡化。SIFT nfeatures=0、nOctaveLayers=3（旧版默认）、contrastThreshold=0.04、edgeThreshold=10、sigma=1.6。BF L2 k=2、严格 Lowe ratio<0.75，取双向 ratio 交集；没有使用 crossCheck=True。恒等几何欧氏距离 ≤3px。种子 3407，OpenCV 线程 1，OpenCL 关闭。', '',
             '固定预算先检测全部点，再按 response 降序；相同响应按 y/x/size/angle/octave/class_id/原索引确定性排序，描述子同步选择。所有条件共享每幅图的一份完整 GT 特征库，GT 不做预算裁剪。一对一约束按 SIFT 特征实例索引验证，GT 特征不会重复计数；同一位置的不同 SIFT 方向仍是不同特征实例，与历史计数保持一致。', '',
             '计数和比例均采用图像级 macro mean/median，保留零匹配、零关键点和低于 K 的图片。比例分母为零记 0，并单独报告零案例；逐图相对变化基线为零时留空。成对相对变化汇总为“均值差/基线均值”，没有对逐图百分比求平均。verified_per_retained 仅作诊断，不能作为独立正确匹配率。', '',
             f"运行版本：Python {config['runtime']['python']} / OpenCV {config['runtime']['opencv']} / NumPy {config['runtime']['numpy']} / Pillow {config['runtime']['pillow']}。路径、参数及全部输入和代码 SHA256 见 config_resolved.json、audit_sources.csv、audit_documents.csv。", '',
             '## 历史 unlimited 回归', '',
             f"状态：**{regression['status']}**。" + (f"逐图数值差异 {regression['difference_count']} 项；最大绝对误差 {regression['maximum_absolute_error']}。" if 'difference_count' in regression else '软件夹具没有研究历史结果。'), '',
             '只有 FULL_PASS 且完成 428 张才能宣称复现旧实验。SUBSET_PASS 仅表示所选 smoke 样本通过。详见 legacy_regression_per_image.csv、legacy_regression_differences.csv 和 legacy_regression.json。', '',
             '## 无限预算', '',
             '| 方法 | N | 平均检测点 | verified 均值 | verified 中位数 | precision 宏均值 | coverage 宏均值 | 零 tentative |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in unlimited:
        lines.append(f"| {row['method']} | {row['image_count']} | {row['mean_detected_total']:.2f} | {row['mean_verified']:.4f} | {row['median_verified']:.2f} | {row['mean_consistency_precision']:.6f} | {row['mean_reference_coverage']:.6f} | {row['zero_tentative_count']} |")
    lines += ['', '一致性 precision 是过滤后 tentative 对应关系的条件比例，可能接近饱和；必须结合计数、覆盖率和零案例解读。', '',
              '## 固定关键点预算（完整列示所有请求的 K）', '',
              '| K | 方法 | verified 均值 | verified 中位数 | coverage 宏均值 | verified/retained 宏均值（诊断） | 实际 retained 均值 | 预算不足 N/比例 |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(fixed, key=lambda r: (int(r['K']), r['method'])):
        lines.append(f"| {row['K']} | {row['method']} | {row['mean_verified']:.4f} | {row['median_verified']:.2f} | {row['mean_reference_coverage']:.6f} | {row['mean_verified_per_retained']:.6f} | {row['mean_retained']:.2f} | {row['budget_shortage_count']}/{percent(row['budget_shortage_fraction'])} |")
    lines += ['', 'detected_total<K 时只保留实际检测数，所有样本仍在分母中；固定预算上限不表示每幅图都达到该预算。每种方法的所有均值、中位数、零案例和不足比例详见 fixed_budget_summary.csv。', '',
              '## 成对差异（绝对值与相对变化）', '',
              '| K | 方法 | 基线 | verified 平均差 | 差值中位数 | 均值相对变化 | 增/平/减 |',
              '|---|---|---|---:|---:|---:|---:|']
    for row in paired:
        lines.append(f"| {row['K']} | {row['method']} | {row['baseline']} | {row['mean_verified_delta']:+.4f} | {row['median_verified_delta']:+.2f} | {percent(row['verified_relative_change_of_means'])} | {row['increased']}/{row['equal']}/{row['decreased']} |")
    lines += ['', '逐图绝对增量、相对变化和零基线案例保存在 paired_differences_per_image.csv；汇总在 paired_differences_summary.csv。', '',
              '## 主指标 verified@256：图像级成对 bootstrap', '',
              '5000 次、有放回抽样图像 ID，每次同一 ID 的两种方法一起抽取，种子 3407，percentile 95% CI（线性分位数）。不同对比共享相同随机抽样序列。不报告 p 值。', '',
              '| 方法 | 基线 | 平均差 | 95% CI | 观察结果 |', '|---|---|---:|---|---|']
    for row in intervals:
        finding = '正向差异' if row['ci95_low'] > 0 else ('负向差异' if row['ci95_high'] < 0 else '区间包含零，未明确区分方向')
        lines.append(f"| {row['method']} | {row['baseline']} | {row['mean_verified_delta']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {finding} |")
    if strongest:
        lines += ['', f"按完整评估范围内的 macro mean verified@256 选取最强已纳入 SOTA：**{', '.join(strongest)}**。若为 smoke 范围，这只是该子集排名。并列方法全部报告，图中仅按名称排序选择代表。最强方法由当前评价集选出；上述区间是固定对比的描述性区间，未做选择偏差或多重比较校正，不能当作额外的确认性检验。"]
    lines += ['', '## 指定样本复核', '',
              '| ID | 方法 | detected_total | retained@256 | verified@256 | verified unlimited |',
              '|---|---|---:|---:|---:|---:|']
    lookup = {(r['image_id'], r['method'], str(r['K'])): r for r in rows}
    for image_id in ['1930', '220', '895', '1757']:
        for method in config['methods']:
            key = (image_id, method['name'], '256')
            if key in lookup:
                row = lookup[key]
                lines.append(f"| {image_id} | {method['name']} | {row['detected_total']} | {row['retained']} | {row['verified']} | {lookup[image_id, method['name'], 'unlimited']['verified']} |")
    lines += ['', 'ID 1757 的 Raw 若仍为 46 个检测点，K=256 时只保留 46，记预算不足，不插值、复制或剔除。图像来源的 20 个场景视觉抽查见配置指定的 source_audit_dir；不把抽查宣称为全体图片的像素级真值认证。匹配坐标另按指定异常样本加固定随机样本独立重算平方距离并检查掩码、一对一约束和恒等矩阵，见 validation.json。', '',
              '## 方法来源说明', '']
    for method in config['methods']:
        if method.get('implementation_note'):
            lines.append(f"- {method['name']}：{method['implementation_note']}")
    lines += ['', '各方法使用已有保存输出；生成这些输出的模型训练流程、检查点选择方式和实现细节由原来源记录说明。SIFT 评价协议统一不意味着各网络训练流程相同，也不将当前结果推广为所有实现或所有数据集上的性能。', '',
              '## 示例图', '']
    if figure['generated']:
        lines += ['qualitative_2x3.png（3.5 英寸宽，300 dpi）和 qualitative_2x3.svg 已生成。列为 Raw / 最强代表 SOTA / ICAR-UIE；行固定 ID 1930、895，选择发生在 V2 统计之前。所有面板画出同一 K=256 上限下实际保留的点，以绿色/品红区分几何一致/未验证，计数直接核对逐图结果。背景为位图，SVG 标签、标记和边框为矢量。没有通过个例替代全部测试集统计。', '', '![qualitative](qualitative_2x3.png)']
    else:
        lines.append('未生成论文图：' + figure['reason'] + '。')
    lines += ['', '## 运行与文件', '',
              '执行入口 tools/sift_reference_consistency_v2.py；使用 --config、--output-dir、--dry-run、--only-ids 和 --K。参考仓库 README_SIFT_V2.md。所有输出目录必须新建，不支持覆盖；本次没有修改原训练测试代码、源图片或 sift_gt_saved。', '',
              '完整性检查见 validation.json、run_status.json；缺失或无效数据见 invalid_missing_sources.csv（成功运行只有表头）。研究结论应依据全部 428 张的表格、成对差值和区间，同时如实陈述正向、无明确差异和负向发现。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
