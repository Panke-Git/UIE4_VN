"""P0 runner: protocol -> Identity gate -> native comparison -> subset AP -> report."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import traceback

from ..utils import load_prepared
from .common import *
from .audit_protocol import audit
from .identity_control import run_identity
from .compare_full_vs_tiled import evaluate_comparison
from .report import write_report


def run(prepared,weights,enhanced,output,uie_run_dir=None,uie_python=None,gpu="0",evaluation_root=None,
        data_root=None,sample_count=32,inventory_only=False,allow_synthetic=False):
    prepared,weights,enhanced = map(lambda p:Path(p).resolve(),(prepared,weights,enhanced))
    protected_outputs = [prepared,weights,enhanced,Path(evaluation_root or prepared.parent/"evaluation")]
    protected_outputs += [Path(p) for p in (data_root,uie_run_dir) if p]
    out = fresh_output(output,protected_outputs)
    runtime_dirs(out)
    status = {"status":"started","command":sys.argv,"detector_python":sys.executable,
              "uie_python":str(uie_python or sys.executable),"sample_count":sample_count,"gpu":gpu,
              "no_training":True,"p1_enabled":False}
    write_json(out/"runner_status.json",status)
    inventory = source_inventory(out)
    diagnostic_sources = [p for p in Path(__file__).parent.glob("*.py")]
    diagnostic_sources += list((PROJECT/"scripts/downstream_duo/diagnostics").glob("*.sh"))
    write_json(out/"diagnostic_source_hashes.json",{str(p):sha256(p) for p in diagnostic_sources})
    availability = {"execution_host":str(PROJECT),"cloud_access_assumed":False,"inputs":{},
                    "given_results_are_not_local_measurements":True}
    roots = [prepared,weights,enhanced,Path(evaluation_root or prepared.parent/"evaluation")]
    if uie_run_dir:
        roots += [Path(uie_run_dir)/"best/best_psnr.pt",Path(uie_run_dir)/"config_resolved.yaml"]
    availability_paths = roots + [Path(evaluation_root or prepared.parent/"evaluation")/c/"validator/predictions.json" for c in ("raw","full")]
    if data_root:
        availability_paths += [Path(data_root),Path(data_root)/"annotations/instances_test.json"]
    for path in availability_paths:
        availability["inputs"][str(path)] = path.exists()
    write_json(out/"availability.json",availability)
    guard = None
    try:
        if inventory_only:
            status["status"] = "inventory_only_real_diagnostics_not_run"
            return out
        meta = load_prepared(prepared,verify=False)
        if meta["synthetic"] and not allow_synthetic:
            raise ValueError("No fake weights/data permitted in real P0; synthetic mode is test-only")
        if not allow_synthetic and sample_count not in (32,64):
            raise ValueError("Production diagnostics predeclare 32 or 64 IDs; no performance-based resampling")
        if data_root or meta.get("source_root"):
            roots += [Path(data_root or meta["source_root"])]
        roots += [Path(r["path"]) for r in inventory]
        print("Hashing protected input trees before diagnostics (may take several minutes)...",flush=True)
        guard = snapshot(roots)
        write_json(out/"input_integrity_before.json",guard)
        a = audit(prepared,weights,enhanced,out/"audit",evaluation_root,data_root,uie_run_dir,allow_synthetic)
        if a["status"] != "pass":
            status["status"] = "stopped_protocol_gate"
            return out
        identity = run_identity(prepared,weights,enhanced,out/"identity",a,gpu)
        if identity["status"] != "pass":
            status["status"] = "stopped_identity_gate"
            return out
        inferred_run = uie_run_dir or a["paths"]["uie_run_dir"]
        uie_python = str(uie_python or sys.executable)
        command = [uie_python,"-m","src.downstream_duo.diagnostics.compare_full_vs_tiled",
            "--phase","infer","--prepared",str(prepared),"--weights",str(weights),"--enhanced",str(enhanced),
            "--output",str(out/"full_vs_tiled"),"--uie-run-dir",str(inferred_run),"--gpu",str(gpu),
            "--audit-json",str(out/"audit/audit_protocol.json"),"--identity-json",str(out/"identity/identity_control.json"),
            "--sample-count",str(sample_count)]
        status["inference_command"] = command
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(out/"runtime/pycache")
        with (out/"full_vs_tiled_inference.log").open("w",encoding="utf-8") as log:
            # Console stays responsive through the line-oriented P0 progress messages.
            process = subprocess.Popen(command,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line)
                log.flush()
            process.wait()
        if process.returncode:
            status.update(status="native_phase_not_completed",inference_returncode=process.returncode)
            return out
        evaluate_comparison(prepared,weights,enhanced,out/"full_vs_tiled",a,identity,gpu)
        status["status"] = "completed_with_explicit_native_eligibility"
    except Exception as error:
        status.update(status="error_or_unavailable",error=repr(error),traceback=traceback.format_exc())
        print(status["error"],flush=True)
    finally:
        if guard:
            print("Checking protected input hashes after diagnostics...",flush=True)
            write_json(out/"input_integrity_after.json",check_snapshot(guard))
        else:
            write_json(out/"input_integrity_after.json",{"status":"not_run","reason":"No real diagnostic inputs were opened for inference/evaluation"})
        write_json(out/"runner_status.json",status)
        summary = write_report(out)
        print(f"P0 status={status['status']}; decision={summary['decision']['code']}; report={out/'diagnosis_summary.md'}",flush=True)
    return out


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--uie-python",default=os.getenv("UIE_PYTHON"))
    parser.add_argument("--sample-count",type=int,default=32,choices=[32,64])
    parser.add_argument("--inventory-only",action="store_true",help="Only availability/source snapshot; no claimed real measurements")
    args = parser.parse_args(argv)
    if not 1 <= args.sample_count <= 64:
        parser.error("sample count must be 1..64")
    out = run(args.prepared,args.weights,args.enhanced,args.output,args.uie_run_dir,args.uie_python,args.gpu,
              args.evaluation_root,args.data_root,args.sample_count,args.inventory_only)
    runner = read_json(out/"runner_status.json")
    if runner["status"] not in ("completed_with_explicit_native_eligibility","inventory_only_real_diagnostics_not_run"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
