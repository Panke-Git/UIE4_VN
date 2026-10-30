# UICF 参数化消融：Git 修改报告

本次只在当前 UIE4_workspace 的 UIE4_VN 子工程操作。没有创建/切换工作区或分支，没有提交或推送。

- 工程：`/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN`
- 开发起点及最终 HEAD：`50b38ce632542cd189db0eb8b5e032a5d2fd6dc1`
- 分支：main；相对已有 origin/main 跟踪引用落后 1 条。未执行 fetch，实际远端状态待人工检查。
- 开发起点 545 个已跟踪文件 SHA256 全部不变；git diff 和 git diff --cached 均为空。
- 原 src/v16、src/v15、src/shared/uicf_inr.py、src/shared/uicf_models.py、configs/config_v16.yaml 保持不变。
- 所有交付代码为新增、未暂存文件；没有修改或删除任何已跟踪文件。

## 全部新增交付文件（完整本地路径）

- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/configs/ablations/uicf_parameterization/direct_residual.yaml`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/configs/ablations/uicf_parameterization/fixed_anchor.yaml`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/configs/ablations/uicf_parameterization/without_anchor.yaml`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/docs/UICF_PARAMETERIZATION_ABLATION_GUIDE_CN.md`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/docs/UICF_PARAMETERIZATION_GIT_REPORT_CN.md`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/docs/UICF_PARAMETERIZATION_LOCAL_TEST_REPORT_CN.md`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/scripts/uicf_parameterization_ablations/test_runs.sh`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/scripts/uicf_parameterization_ablations/train_sequential.sh`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/scripts/uicf_parameterization_ablations/verify_local.sh`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/__init__.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/capacity.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/engine.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/models/__init__.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/models/network.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/protection_snapshot.json`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/protocol.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/smoke.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/test.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/train.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/tests/uicf_parameterization_ablations/__init__.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/tests/uicf_parameterization_ablations/conftest.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/tests/uicf_parameterization_ablations/test_models.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/tests/uicf_parameterization_ablations/test_workflow.py`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/tools/uicf_parameterization_report.py`

## 生成的本地检查与结果文件

- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/parameter_counts.json`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/results.md`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/results.csv`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/local_tests/new_tests.xml`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/local_tests/regressions.xml`
- `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/local_tests/cli_smoke.log`
- 人工数据/临时 checkpoint：`/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/local_tests/tmp`、`/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/cli_smoke`。
- 上述 analysis 输出沿用原 .gitignore 排除，不能把人工 smoke 的指标作为论文分数提交。

## 当前 Git 状态

```text
## main...origin/main [behind 1]
?? configs/ablations/uicf_parameterization/direct_residual.yaml
?? configs/ablations/uicf_parameterization/fixed_anchor.yaml
?? configs/ablations/uicf_parameterization/without_anchor.yaml
?? docs/UICF_PARAMETERIZATION_ABLATION_GUIDE_CN.md
?? docs/UICF_PARAMETERIZATION_GIT_REPORT_CN.md
?? docs/UICF_PARAMETERIZATION_LOCAL_TEST_REPORT_CN.md
?? scripts/uicf_parameterization_ablations/test_runs.sh
?? scripts/uicf_parameterization_ablations/train_sequential.sh
?? scripts/uicf_parameterization_ablations/verify_local.sh
?? src/uicf_parameterization_ablations/__init__.py
?? src/uicf_parameterization_ablations/capacity.py
?? src/uicf_parameterization_ablations/engine.py
?? src/uicf_parameterization_ablations/models/__init__.py
?? src/uicf_parameterization_ablations/models/network.py
?? src/uicf_parameterization_ablations/protection_snapshot.json
?? src/uicf_parameterization_ablations/protocol.py
?? src/uicf_parameterization_ablations/smoke.py
?? src/uicf_parameterization_ablations/test.py
?? src/uicf_parameterization_ablations/train.py
?? tests/uicf_parameterization_ablations/__init__.py
?? tests/uicf_parameterization_ablations/conftest.py
?? tests/uicf_parameterization_ablations/test_models.py
?? tests/uicf_parameterization_ablations/test_workflow.py
?? tools/uicf_parameterization_report.py
```

## 人工后续操作

选择性暂存清单、提交/推送和 WSL 安全同步见中文运行指南第二阶段。先处理 main 相对已有远程跟踪引用落后 1 条的问题；不要强制覆盖本地或 WSL 修改。保护快照指向开发起点，之后上游合并对旧文件的合法修改需要另行审查，不自动改写快照。
