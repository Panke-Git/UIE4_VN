"""Chinese evidence report with explicit unavailable-data and manuscript limits."""
from __future__ import annotations

import json
from pathlib import Path

from . import combined_status
from .io import write_csv, write_json


def fmt(value):
    if value is None or value == "":
        return "—"
    return format(value, ".12g") if isinstance(value,float) else str(value)


def table(headers,rows):
    escape=lambda value:fmt(value).replace("|","\\|").replace("\n"," ")
    return ["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|",
            *["| "+" | ".join(escape(v) for v in row)+" |" for row in rows]]


def manuscript(audit,path,spatial):
    if path is None:
        audit.check("manuscript","paper_text","UNVERIFIED",
            "PENDING_MANUSCRIPT：需要当前 Section 4.4 全文、Supplementary Table 4 及对应表注/图注。")
        return dict(status="PENDING_MANUSCRIPT",text=None,source=None,suggestions=[])
    path=Path(path)
    kind="json" if path.suffix.lower()==".json" else "text"
    content=audit.read(path,kind,section="manuscript")
    if content is None:
        return dict(status="PENDING_MANUSCRIPT",text=None,source=str(path),suggestions=[])
    text=json.dumps(content,ensure_ascii=False) if isinstance(content,dict) else content
    # Reading arbitrary prose is not automated scientific semantic verification.
    audit.check("manuscript","semantic_review","UNVERIFIED",
        "已保留论文输入原文及 SHA256；需结合本报告人工审阅科学语义，不能仅凭关键词宣称无需实质修改。",[path])
    suggestions=[]
    if isinstance(content,dict):
        aliases={"anchor_deviation":"anchor_deviation","image_gradient":"image_gradient",
                 "shifted_raw_field":"shifted_raw_field","raw_field":"raw_field"}
        for row in content.get("table_4_rows",[]):
            name=aliases.get(row.get("response"))
            for metric in ("spearman","iou"):
                matches=[r for r in spatial if r["response"]==name and r["metric"]==metric and r["observed_mean"] is not None]
                actual=row.get(metric)
                if name is None or actual is None or not matches:
                    audit.check("manuscript","table_numeric","UNVERIFIED",f"无法核对 {row}",[path])
                    continue
                from .statistics import equal_number
                mismatches=[r for r in matches if not equal_number(r["observed_mean"],actual,0.0005+1e-12)]
                audit.check("manuscript","table_numeric","FAIL" if mismatches else "PASS",
                            f"{name}.{metric}: 论文={actual}; 重算={[r['observed_mean'] for r in matches]}",[path],"P0")
                if mismatches:
                    suggestions.append(dict(location="Supplementary Table 4",current=str(row),
                        problem="论文记录与逐图重聚合值不在三位小数容差内",
                        suggestion="先核对来源，再按对应原始 CSV 修正数值；不改原始结果",severity="P0"))
    return dict(status="UNVERIFIED",text=text,source=str(path),suggestions=suggestions)


def claim_matrix(alignment_results,intervention_results):
    spatial=[r for result in alignment_results for r in result.get("records",[])]
    raw=[r for r in spatial if r["response"]=="raw_field" and r["metric"]=="spearman"]
    b_status=combined_status([r.get("status","UNVERIFIED") for r in alignment_results])
    c_status=combined_status([r.get("status","UNVERIFIED") for r in intervention_results])
    raw_positive=bool(raw) and all(r["observed_mean"] is not None and r["observed_mean"]>0 for r in raw)
    intervention_records=[r for item in intervention_results for r in item.get("records",[]) if r["variant"]!="original"]
    positive=[f"{r['variant']}.{r['metric']}" for r in intervention_records if r["paired_gain"] is not None and r["paired_gain"]>0]
    negative=[f"{r['variant']}.{r['metric']}" for r in intervention_records if r["paired_gain"] is not None and r["paired_gain"]<=0]
    return [
        dict(evidence="A",scientific_question="为何采用该校正参数化？",experiment="Correction Parameterization Ablation",
             supported_claim="比较 I+Delta、I+R*I、I+R*(I-0.5)、I+R*(I-b) 在同一训练协议下的经验恢复效果。",
             limitation="任务①按用户说明已完成，本工具不重算其结果，也不据此宣称严格参数匹配、数学表达能力更强或普遍最优。",
             status="UNVERIFIED",source="任务①已完成的用户说明；本次未重审数值"),
        dict(evidence="B",scientific_question="系数场学到了怎样的空间组织？",experiment="Spatial Correspondence",
             supported_claim=("已读取数字显示系数幅度与参考恢复残差代理的正空间相关；仍需满足来源、完整性和协议检查。"
                              if raw_positive else "需以已存在逐图数字检验系数幅度与参考恢复需求代理的空间对应；当前不能预填积极结论。"),
             limitation="仅针对 ||R||_2；不是实际校正幅度结论，不是真实物理退化场或物理因果证明。",
             status=b_status,source="; ".join(r["directory"] for r in alignment_results)),
        dict(evidence="C",scientific_question="已有组织是否影响冻结模型的恢复？",experiment="Functional Intervention",
             supported_claim=f"应逐干预判断冻结模型的功能依赖；已读取正向性能差={positive}；非正差={negative}。",
             limitation="正差只是点估计，CI 跨零时需保留不确定性；分布外干预也可能影响性能。不是重新训练消融、跨种子稳定性、物理因果识别或所有网络中的不可替代性证明。",
             status=c_status,source="; ".join(r["directory"] for r in intervention_results)),
    ]


def render(output,audit,summary,alignments,interventions,paper,claims):
    spatial=[r for result in alignments for r in result.get("records",[])]
    functional=[r for result in interventions for r in result.get("records",[])]
    write_csv(output/"spatial_correspondence_audit.csv",spatial,
        ["analysis_dir","response","metric","field","observed_mean","saved_mean","valid_count",
         "missing_count","invalid_count","expected_count","provided_table_value","table_atol",
         "numeric_status","table_status","audit_status","definition","source_csv"])
    write_csv(output/"functional_intervention_audit.csv",functional,
        ["analysis_dir","variant","metric","observed_mean","saved_mean","valid_count","invalid_count",
         "original_mean","paired_count","paired_gain","gain_direction","ci95_low","ci95_high",
         "numeric_status","audit_status","source_csv"])
    write_csv(output/"evidence_claim_matrix.csv",claims,
        ["evidence","scientific_question","experiment","supported_claim","limitation","status","source"])
    source_rows=list(audit.sources.values())
    write_json(output/"audit_summary.json",summary)
    write_json(output/"source_manifest.json",dict(
        source_files=source_rows,
        user_reference_policy="任务提供的数字仅作核对阈值，不作为重算结果；无源文件 SHA256，不能代替实验证据。",
        statistics="对有效逐图数值等权宏平均；配对按 sample_id；Bootstrap 重采样单位为图像。",
        default_checkpoint_policy="不反序列化、不导入模型；默认仅 stat checkpoint，可选只流式 SHA256。",
        audit_definition="source status 是读取状态；verification_status 是其关联检查的最严格状态；两者不等同。"))
    lines=["# UICF 实验证据链核验报告",
        "",f"生成时间：{summary['created_at']}；审计状态：**{summary['status']}**。", "",
        "本报告只核验已存在的数值与元数据，不训练、不运行模型推理、不重新生成特征图。"
        "PASS 表示相应检查范围内证据充分且一致；FAIL 表示明确冲突；UNVERIFIED 表示缺失或不足。", "",
        "## 一、实验来源", "",
        f"- 唯一基线：{summary['run_dir']}",
        f"- Best Validation PSNR checkpoint epoch：{fmt(summary['baseline']['checkpoint_epoch'])}",
        f"- checkpoint 身份依据：{summary['baseline']['checkpoint_hash_basis']}",
        f"- 原始模型/划分：V16；LSUI19；3466 / 385 / 428；seed 3520；200 epochs。",
        "- 以上是目标协议，是否实际满足须以 audit_summary.json 中逐项检查为准。",
        "- 每个实际读取文件的完整路径、SHA256、字段、行数和状态见 source_manifest.json。",
        "- 当前源码逻辑审阅与历史运行版本绑定分开报告；缺少历史源码指纹不能由当前代码代替证明。", "",
        "## 二、Spatial Correspondence", "",
        "以下定义来自已审阅的源码版本；历史运行是否使用该版本，另查历史源码指纹与协议状态。"
        "当前源码匹配不能补全缺失的历史运行记录。", "",
        "对象为 D_R=||R||_2，参考为 D_ref=||Y-I||_2。先在像素上求 RGB L2 范数，"
        "再对标量幅度图做不重叠 16×16 平均；不是先池化 RGB 向量再求范数。"
        "256×256 图像形成 16×16=256 个块，Top-20% 精确选 ceil(256×0.2)=52 块。", "",
        "Spearman 使用平均并列秩后计算 Pearson；常量图对应未定义值，不强填为零。"
        "Top-20% 使用稳定降序排序，边界并列时按展平索引确定选中项；IoU=交集/并集。"
        "先逐图计算，再对有效图像等权平均，不按全部像素微平均。", "",
        "Anchor deviation 使用本图预测 b 的 ||I-b||_2。Image gradient 使用 RGB 各通道 3×3 Sobel /8、"
        "reflect 边界，并计算 sqrt(sum_RGB(gx²+gy²))，再池化；不使用灰度图。", "",
        "相关性分析中的 Shifted Field 对池化后的 ||R||_2 标量图做循环位移。"
        "位移由 NumPy SeedSequence([null_seed,sample_index]) 生成，排除零位移，"
        "要求至少一个轴位移不小于块网格该轴的四分之一；实际次数和种子见协议。"
        "先对每图有效位移结果求均值，再跨图等权平均。它保留池化标量图的数值集合，"
        "不能说工具直接对原始 RGB R 运行了该位移；对整块对齐的位移，范数/池化后位移有相应等价关系。", "",
        "下表的“任务参考”来自用户提供的待核验 Table 4/历史记录，"
        "不是已读取论文或缺失实验的替代值。逐图重算缺失时保留空值。", ""]
    lines+=table(["分析目录","对象","指标","逐图重算","保存均值","有效 n","任务参考","数字核对","来源/协议"],
        [(r["analysis_dir"],r["response"],r["metric"],r["observed_mean"],r["saved_mean"],r["valid_count"],
          r["provided_table_value"],r["numeric_status"],r.get("audit_status")) for r in spatial])
    lines+=["","## 三、Functional Intervention","",
        "保持 I、同一次预测的 b、参考 Y 和 Backbone/QGRD 不变，仅将 R 替换为 R_tilde："
        "Ic_int=I+R_tilde*(I-b)，再经原 Backbone/QGRD 输出。中间 Ic 不 clamp；"
        "最终输出转 float32 并 clamp 后计算原 V16 指标，不从 PNG 反读计算。", "",
        "- original：使用未经干预的诊断完整前向输出，并检查与普通完整前向/Backbone 路径一致。",
        "- spatial_mean：每图、每通道独立求 H/W 均值并广播，保留各通道均值，去除空间变化。",
        "- shift_xy / shift_y / shift_x：对 RGB R 循环位移 H//4、W//4；256×256 时为 64 像素。"
        "这是固定的单次功能干预，不是相关性分析中的多位移零对照。",
        "- channel_cycle_1：新 RGB=旧 G,B,R；channel_cycle_2：新 RGB=旧 B,R,G。仅置换 R，不置换 I 或 b。",
        "- zero_field：冻结模型内关闭 UICF 校正，不等同于从头训练 w/o UICF。",
        "- 核对 eval/inference-mode 的已审阅源码、历史源码指纹、前后参数/缓冲 digest、诊断值及完成标记。"
        "只读记录不能重新证明未保存的运行细节。", "",
        "PSNR/SSIM 的配对差=Original−Intervention；ΔE00 的配对差=Intervention−Original。"
        "正数统一表示 original 更好。配对只按 sample_id，缺失配对不会按行序误配。"
        "Bootstrap 复核同一 checkpoint 的图像级配对差；使用记录的种子、次数、"
        "NumPy PCG64 和 2.5%/97.5% percentile，区间未做多重比较校正，不表示跨训练种子稳定性。",""]
    lines+=table(["分析目录","干预","指标","均值","内部 Original","正向配对差","配对 n","CI low","CI high","核验"],
        [(r["analysis_dir"],r["variant"],r["metric"],r["observed_mean"],r["original_mean"],r["paired_gain"],
          r["paired_count"],r["ci95_low"],r["ci95_high"],r.get("audit_status")) for r in functional])
    lines+=["","### 内部 Original 与官方测试的差异","",
        "下列差异只诊断复现情况，干预配对差不使用官方摘要作替代基线。"
        "缺少 AMP、评估版本等元数据时，即使数字接近，也不能宣称协议已完全一致。",""]
    lines+=table(["目录","指标","内部 Original","官方 CSV 均值","内部−官方","最大逐图差","配对 n"],
        [(r["directory"],metric,v["original_mean"],v["official_mean"],v["original_minus_official"],
          v["max_per_image_error"],v["paired_count"]) for r in interventions for metric,v in r.get("official_comparison",{}).items()])
    lines+=["","## 四、实际校正量及解释边界","",
        "R 是系数场；实际校正 Delta I=Ic−I=R*(I−b)。"
        "||R||_2 的正相关不能证明 ||Delta I||_2 与参考恢复需求高度相关。"
        "两者的重算值均保留在第二部分，不删去不利值。若真实记录重现接近零或负相关，"
        "它限制的是实际校正幅度的论断，不能直接否定系数表示空间组织的独立分析价值。", "",
        "D_ref=||Y−I||_2 是配对参考恢复需求代理，不能直接等同于真实水下物理退化场。", "",
        "## 五、论文结论与三类证据","",
        f"论文文字核验状态：**{paper['status']}**。"]
    if paper["text"] is not None:
        lines+=["","### 已提供论文原文（未自动改写）","","~~~text",paper["text"],"~~~"]
    else:
        lines+=["","尚缺 Section 4.4 全文、Supplementary Table 4、对应表注/图注。"
                "不能据任务描述虚构原句，也不能宣称 No substantive revision required。"]
    lines+=["","以下为独立证据的分工，不是严格的物理因果证明链：",""]
    lines+=table(["Evidence","科学问题","实验","可支持的命题","边界","状态"],
        [(r["evidence"],r["scientific_question"],r["experiment"],r["supported_claim"],r["limitation"],r["status"]) for r in claims])
    lines+=["","### 有原文依据的最小修改建议",""]
    lines+=table(["位置","当前表述","问题","建议","严重程度"],
        [(r["location"],r["current"],r["problem"],r["suggestion"],r["severity"]) for r in paper["suggestions"]])
    if not paper["suggestions"]:
        lines+=["","当前没有可据真实原文列出的修改项；这不等于已确认论文无需修改。"]
    lines+=["","### 当前可以明确的判断","",
        "- 第 4.4 节：是否需要修正具体原句取决于论文原文；避免混用 R 与 Delta I、物理因果性和跨训练种子稳定性。",
        "- Table 4：数字是否一致见逐图重算与独立舍入容差检查；保留既有结构。",
        "- 不自动要求在 Table 4 新增 Actual Correction 一行。若表题/表注已明确是系数表示，"
        "实际校正量结果不构成必须扩表的理由。",
        "- 系数相关性、冻结模型干预及参数化消融可互补，但不证明所有架构中的必要性或数学表达能力更强。", "",
        "## 六、最终状态、缺口与最小补充核验",""]
    lines+=table(["范围","状态"],summary["section_status"].items())
    issues=[c for c in audit.checks if c["status"]!="PASS"]
    lines+=["",f"最终状态：**{summary['status']}**；未通过/未确证检查 {len(issues)} 项。"
            "完整逐项结果和未舍入数字见 audit_summary.json。",""]
    lines+=table(["范围","检查","状态","问题或缺失","严重程度"],
        [(c["section"],c["check"],c["status"],c["detail"],c["severity"]) for c in issues])
    lines+=["","优先补齐现有文件、历史源码/命令与 checkpoint 身份记录。"
        "若唯一缺口是 checkpoint 二进制身份，可人工加 --hash-checkpoint 做 CPU 文件哈希，"
        "不需要 GPU 或加载模型。论文原文缺失时补充原文。"
        "如果历史产物未保存某项关键运行事实，保留 UNVERIFIED，先查原日志/版本记录；"
        "本工具不自动触发任何补充实验。",""]
    (output/"EVIDENCE_CHAIN_AUDIT_CN.md").write_text("\n".join(lines),encoding="utf-8")
