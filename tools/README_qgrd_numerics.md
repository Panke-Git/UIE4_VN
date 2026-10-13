# A2 / Shared Query 数值异常诊断

注意：正式 A2 runner 已新增固定 FP32 评价协议。下面的旧 Epoch 123 诊断记录
对应修复前的源码。更新 A2 包后，旧 run 会因缺少 FP32 声明或源码 hash 不同而被拒绝；
不要为重复诊断修改旧 config/checkpoint 或删除校验。旧诊断应使用原源码 checkout。
对新 FP32 run，工具中的 configured_amp 是独立的 AMP 对照探针，不是正式评价协议；
正式验证/测试请使用新 A2 train/test 入口。

本工具定位已有 A2 训练在 validation 中出现 NaN/Inf 的问题。
它不是新的正式训练入口，不是自动稳定化修复，也不生成论文指标。

只新增 tools/diagnose_qgrd_numerics.py 与独立测试；不修改 shared、A1–A4、
V16/V19、配置、清单、checkpoint 或 run metadata。工具位于 A2 包之外，
因此不改变旧 run 保存的包源码 SHA256，不需要重新 pin 或修改旧 checkpoint。

## 部署

将本地新增的 tools/diagnose_qgrd_numerics.py 上传到服务器 UIE4_VN/tools/ 的同名位置。
不要用修改过的 A2 包覆盖原源码。复用当前 pax 环境，无需新增依赖。

## 针对本次 Epoch 124 失败的命令

在服务器项目根目录：

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN

python tools/diagnose_qgrd_numerics.py \
  --run-dir experiments/qgrd_ablations/shared_query/lsui/qgrd_shared_query_UICF_ColorQuery_SharedQuery_seed3520_20261013_001625_927546 \
  --checkpoint last \
  --expect-checkpoint-epoch 123 \
  --gpu 0 \
  --replay-next-epoch \
  --log-every 10
```

只重放 Epoch 124，不训练后续 125–200。重放在内存中进行，诊断额外检查会增加耗时。
不要同时对该 run 执行正式 resume。工具也不改变 learning rate、batch size、
augmentation、precision、loss、gradient clipping、模型参数注册或数学公式。

若数据迁移，追加 --data-root /absolute/path，必须是同一组图像与配对。

保存模型、optimizer、scaler、Python/NumPy/Torch/CUDA RNG 和 DataLoader generator
均按原生 resume 顺序恢复。使用同一个 generator 构造原 batch/worker 配置的 Train/Validation loader。
不改 seed、worker 数或输入尺寸来“避免”故障。原 deterministic=false 会保留；
CUDA kernel 的非确定性以及环境差异可能使本次没有复现原异常。

## 其他模式

仅检查 checkpoint，不需要图像，也不训练：

```bash
python tools/diagnose_qgrd_numerics.py --run-dir "<A2_RUN_DIR>" --checkpoint last
```

检查旧 checkpoint 的验证 forward，不更新权重：

```bash
python tools/diagnose_qgrd_numerics.py \
  --run-dir "<A2_RUN_DIR>" --checkpoint last --gpu 0 --validate-checkpoint
```

--validation-batches 0 默认检查整个冻结 Validation，遇到首个数值问题停止。
可显式设为 1 检查第一批，但“第一批正常”不等于完整验证正常。
不加载 Test 图像，不运行 native test。

CPU 模式两次验证都是 FP32，会明确警告；不能据此认定 CUDA AMP 故障已经解决。

## 实际检查

- checkpoint epoch/version/model type、source SHA256、resolved_config、冻结三集合协议。
- model parameters/floating buffers 与 optimizer state 的 NaN/Inf。
- AMP scaler metadata，拒绝非有限值或非正 scale。
- max_abs Top-30、所有 BatchNorm running_mean/running_var/eps，以及负 running_var。
- base_queries 和四级 guidance 参数幅值。
- 每个模块输入/输出的 finite 检查，覆盖 UICF、encoder、Query 链、guide query_projection、
  attention、output_projection、Local Refine、最终输出。
- 最早观察到的异常包含 phase/step/sample IDs、tensor 名、shape/dtype/device、
  NaN/+Inf/−Inf 数量及 finite-only min/max/abs_max/mean/std。
- 全部非有限时，finite-only 统计为 null，不因诊断本身再次计算出 NaN。
- 训练 backward 后观察 scaled gradients，不增加额外 unscale 或 optimizer step。
  GradScaler 原有的正常跳步不被诊断当作必须中断训练的条件。
- 每次 optimizer step 后检查参数、buffer 和 optimizer；这只是首次观察到的状态，
  不等于已经证明异常由 optimizer 而非该步 forward/BatchNorm 更新引起。
- 验证从 loader 读取一份 batch，同一权重分别运行原配置 AMP 与 FP32。
  eval 下不更新 optimizer 或 BatchNorm running statistics。

hooks 只属于外部诊断工具，不是正式模型实现。它们不修改返回 tensor、
不替换模块、不抛出异常阻止原计算、不改变 gradient flow。
统计中的 detach 仅用于读取数值，不用于模型前向或 loss。

“首次异常”指最早观察到的模块边界，不声称已定位 fused attention kernel 内部的每个原语。
同一 batch 比较 AMP/FP32 仍可能使用不同 backend kernel；AMP-only 结果是定位证据，
不是“改一个 flag 就一定能永久修好”的证明。

## 输出与下一步

默认仅更新：

```text
diagnostics/qgrd_numerics/<run_name>/replay_next_epoch.json
```

inspect_only / validate_checkpoint 模式分别使用独立的固定报告名。
可用 --output-json 指定实验/源码目录之外的 JSON 路径。
报告以临时文件完成后原子替换，不生成多份时间戳结果；工具拒绝输出到
experiments、src、configs、split、tests、tools 或原 run 内。

原实验目录不写入，selected checkpoint 字节也不修改；最终核对目录文件 size/mtime 和 checkpoint SHA256。
诊断失败也可能生成 ERROR 报告，但不会修改训练状态、日志、指标或 checkpoint。

终端关键输出：

- FIRST_NONFINITE_BOUNDARY：最早观察到的模块边界及数值统计。
- FIRST_POST_STEP_STATE_ISSUE：训练中首次观察到参数/buffer/optimizer 异常。
- DIAGNOSIS_STATUS：诊断结论。
- REPORT_JSON：完整 JSON 路径。

| Status | 含义 |
|---|---|
| CHECKPOINT_FINITE | 仅检查模式未发现 checkpoint 状态异常 |
| CHECKPOINT_STATE_INVALID | checkpoint 参数、buffer、optimizer 或 BN variance 已异常 |
| POST_TRAIN_STATE_INVALID | 重放训练后状态异常，检查首个异常 step/module 和 BN 报告 |
| TRAINING_FORWARD_FAILURE | 重放时先在 training 出现原有限值检查失败，未完成整轮 |
| AMP_ONLY_FORWARD_FAILURE | 同一验证 batch，原 CUDA AMP 失败而 FP32 输出/loss 有限 |
| FP32_FORWARD_FAILURE | FP32 也失败，不能只归因于 AMP |
| INTERMEDIATE_NONFINITE_WITH_FINITE_OUTPUT | 中间 tensor 非有限，但最终输出被后续操作变成有限值 |
| VALIDATION_RUNTIME_ERROR | 模式中有算子异常，检查 runtime error；工具也尝试另一模式 |
| NOT_REPRODUCED | 本次 CUDA 重放/检查未观察到异常，不代表长期稳定性已证明 |
| NO_FAILURE_OBSERVED_CPU_OR_FP32_ONLY | 没有实际检验 CUDA AMP |
| ERROR | 配置、源码、数据、scaler metadata、OOM 等执行错误，检查 traceback |

预期数值问题完成诊断后退出码为 2，正常检查为 0，执行错误为 1；
退出码 2 不表示它破坏了原 run。请把完整 JSON 发回再选择最小修复策略。
不要直接改 A2 源码后删除 hash 校验强行恢复，也不要用 nan_to_num/clamp 掩盖问题。

## 本地测试

```bash
python -m pytest tests/qgrd_ablations/test_numerics_diagnostic.py -q
```

测试覆盖统计边界、NaN/Inf 注入、FP16 cast 溢出、hooks 的 output/gradient/RNG 不变、
原生 resume 与观察性重放的权重/optimizer/RNG 精确一致、原 run 字节不变、
AMP/FP32 结果分类、BN 注入、运行错误与另一模式尝试、输出路径保护和原子报告。

这些 CPU 合成测试不是云端 Epoch 124 已复现或数值训练故障已修复的证明。

本次实际本地结果：诊断专项 19 passed；全仓库 403 passed，0 failed/skipped。
全仓库旧可视化测试使用项目外的临时 Matplotlib 依赖目录运行，未修改 requirements 或全局环境。
任务开始时的 459 个已跟踪文件 SHA256 全部不变；只新增本工具、此说明和诊断测试。
当前本地没有 CUDA，也没有云端真实 A2 checkpoint，因此实际 Epoch 124 根因仍待上述云端命令确认。
