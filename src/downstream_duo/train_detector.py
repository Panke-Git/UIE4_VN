"""Train one detector on prepared Raw train; select best.pt using Raw val only."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

from .utils import (OUTPUT, fresh_dir, load_config, load_prepared, output_path, pair_identity,
                    require_yolo, sha256, versions, write_json, write_yaml)


def train(prepared, out, config, model_path=None, device=None, batch=None, synthetic=False):
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    meta = load_prepared(prepared)
    if bool(meta["synthetic"]) != synthetic:
        raise ValueError("Synthetic smoke must be explicitly isolated from real DUO")
    YOLO = require_yolo()
    kwargs = dict(config["train"])
    model_path = model_path or kwargs.pop("model")
    kwargs.pop("model", None)
    if not synthetic and (str(model_path).endswith(".yaml") or not str(model_path).endswith(".pt")):
        raise ValueError("Real experiment needs COCO-pretrained YOLO11s .pt weights")
    if not Path(model_path).is_file() and model_path != "yolo11s.pt" and not synthetic:
        raise FileNotFoundError("Only yolo11s.pt may be auto-downloaded; supply offline .pt explicitly")
    if model_path == "yolo11s.pt" and not Path(model_path).is_file():
        model_path = OUTPUT / "pretrained/yolo11s.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
    if device is not None:
        kwargs["device"] = device
    if batch is not None:
        kwargs["batch"] = batch
    if kwargs.get("resume"):
        raise ValueError("This protocol refuses automatic retraining/resume")
    if kwargs.get("amp"):
        raise ValueError("Use amp=false: avoid Ultralytics AMP self-check downloading an extra model")
    if not synthetic and (kwargs["seed"] != meta["seed"] or kwargs["imgsz"] != 640):
        raise ValueError("Production seed must match split, imgsz must be 640")
    fresh_dir(out)
    # Construct trusted paths here: no caller can point training/validation at official test.
    train_yaml = out / "raw_train_val.yaml"
    write_yaml(train_yaml, {"path": str(prepared), "train": "images/train_raw", "val": "images/val_raw",
                            "names": meta["mapping"]["names"]})
    lock = prepared / "detector_training_lock.json"
    registry = {"status": "started", "synthetic": synthetic, "prepared": str(prepared), "output": str(out),
                "identity": pair_identity(meta), "train_manifest_sha256": sha256(prepared / "manifests/train.tsv"),
                "val_manifest_sha256": sha256(prepared / "manifests/val.tsv"), "versions": versions(),
                "evaluation": config["evaluation"], "visualization": config["visualization"],
                "train": kwargs, "initial_model": str(model_path), "started_unix": time.time()}
    # Exclusive creation protects against accidental second training and concurrent launch.
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(registry, stream, indent=2)
    write_json(out / "protocol.json", registry)
    write_yaml(out / "train_config.yaml", {**kwargs, "model": str(model_path), "data": str(train_yaml)})
    from ultralytics.utils import LOGGER
    handler = logging.FileHandler(out / "train.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        model = YOLO(str(model_path), task="detect")
        if not synthetic:
            from ultralytics.nn.tasks import yaml_model_load
            expected = yaml_model_load("yolo11s.yaml")
            if model.model.yaml.get("backbone") != expected["backbone"] or model.model.yaml.get("head") != expected["head"] or model.model.yaml.get("scale") != "s" or model.model.yaml.get("nc") != 80:
                raise ValueError("Production initialization must be COCO 80-class YOLO11s")
        initial = Path(getattr(model, "ckpt_path", None) or str(model_path))
        registry["initial_model_sha256"] = sha256(initial) if initial.is_file() else None
        # Disable AMP self-check model downloads; full precision is an explicit logged option.
        model.train(data=str(train_yaml), project=str(out), name="run", exist_ok=False,
                    **kwargs)
        best = out / "run/weights/best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"Training produced no best.pt: {best}")
        registry.update(status="complete", weights=str(best.resolve()), weights_sha256=sha256(best),
                        finished_unix=time.time())
    except Exception as error:
        registry.update(status="failed", error=repr(error))
        LOGGER.exception("DUO detector training failed")
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
        write_json(lock, registry)
        write_json(out / "protocol.json", registry)
    return registry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default=str(OUTPUT / "prepared_duo"))
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--model", help="Offline COCO-pretrained YOLO11s .pt, or first-run yolo11s.pt")
    p.add_argument("--device")
    p.add_argument("--batch", type=int)
    args = p.parse_args(argv)
    result = train(args.prepared, output_path(args.output), load_config(args.config), args.model, args.device, args.batch)
    print(result["weights"], result["weights_sha256"])


if __name__ == "__main__":
    main()
