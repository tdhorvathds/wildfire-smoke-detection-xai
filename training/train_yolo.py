# ===============================
# Batch Runner for YOLO Training
# ===============================
import sys
import time
import csv
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch

# -------------------------
# Paths
# -------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "training"
CSV_PATH = PROJECT_DIR / "configs" / "yolov11" / "yolo_batch_example.csv"
BATCH_DIR = PROJECT_DIR / "results" / "trial"
BATCH_LOG = BATCH_DIR / "experiment_summary.csv"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROJECT = "outputs/predictions"

# -------------------------
# Helpers
# -------------------------
def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def rowval(row, key, default=None):
    v = row.get(key, default)
    if pd.isna(v):
        return default
    return v

def resolve_run_dir(project_value, name_value, project_root):
    p = Path(str(project_value))
    if not p.is_absolute():
        p = project_root / p
    return (p / str(name_value)).resolve()

def checkpoint_status(run_dir):
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    return {"best_exists": best.is_file(), "last_exists": last.is_file()}

def job_completed(run_dir):
    s = checkpoint_status(run_dir)
    return s["best_exists"] and s["last_exists"]

def build_name(row):
    variant = rowval(row, "model_variant", f"{rowval(row,'model_family','yolov11')}{rowval(row,'model_size','s')}")
    aug = "aug" if str(rowval(row, "aug", "")).lower() in {"1","true","t","yes","y"} else "noaug"
    ws_mode = str(rowval(row, "weight_sampling", "false")).lower().strip()
    if ws_mode == "static":
        ws = "ws_static"
    elif ws_mode == "dynamic":
        ws = "ws_dynamic"
    else:
        ws = "nows"
    return f"{variant}_{aug}_{ws}"

def ensure_columns(df):
    defaults = {
        "job_id": range(1, len(df)+1),
        "model_family": "yolov11",
        "model_size": "s",
        "aug": False,
        "optimizer": "SGD",
        "lr0": 0.0005,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "scheduler": "cosine",
        "epochs": 50,
        "patience": 6,
        "imgsz": 640,
        "batch": 32,
        "num_workers": 16,
        "data": "data/pyro-sdis/splits_3fold/fold3/data_fold3.yaml",
        "label_smoothing": 0.05,
        "weight_sampling": "false",
        "project": DEFAULT_PROJECT,
        "seed": 42,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
    if "name" not in df.columns:
        df["name"] = [build_name(df.iloc[i]) for i in range(len(df))]
    if "model_variant" not in df.columns:
        df["model_variant"] = df.apply(
            lambda r: f"{r.get('model_family','yolov11')}{r.get('model_size','s')}",
            axis=1
        )
    return df

# -------------------------
# Logging
# -------------------------
def log_result(row, status, start_t, end_t, err_msg="", run_dir=None):
    """Log job results with optional augmentation + sampling info."""
    aug_yaml_used = ""
    sampling_mode = ""
    if run_dir and (run_dir / "run_config.json").exists():
        try:
            cfg = json.loads((run_dir / "run_config.json").read_text())
            aug_yaml_used = cfg.get("aug_yaml_used", "")
            sampling_mode = cfg.get("weight_sampling", "")
        except Exception:
            pass

    rec = {
        "job_id": rowval(row, "job_id", ""),
        "name": rowval(row, "name", ""),
        "optimizer": rowval(row, "optimizer", ""),
        "lr0": rowval(row, "lr0", ""),
        "epochs": rowval(row, "epochs", ""),
        "weight_sampling": sampling_mode or rowval(row, "weight_sampling", ""),
        "label_smoothing": rowval(row, "label_smoothing", ""),
        "aug_yaml_used": aug_yaml_used,
        "status": status,
        "error": err_msg,
        "gpu_name": gpu_name(),
        "started_at": datetime.fromtimestamp(start_t).strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": datetime.fromtimestamp(end_t).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round((end_t - start_t) / 60.0, 2),
    }
    write_header = not BATCH_LOG.exists()
    with BATCH_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if write_header:
            w.writeheader()
        w.writerow(rec)

# -------------------------
# Command builder
# -------------------------
def yolo_cmd(row):
    """Build CLI command for yolo_core.py."""
    cmd = [
        sys.executable, str(SRC_DIR / "yolo_core.py"),
        "--model_family", str(rowval(row, "model_family", "yolov11")),
        "--model_size", str(rowval(row, "model_size", "s")),
        "--model_variant", str(rowval(row, "model_variant", "")),
        "--use_aug", str(rowval(row, "use_aug", False)),
        "--optimizer", str(rowval(row, "optimizer", "SGD")),
        "--lr0", str(rowval(row, "lr0", 0.005)),
        "--momentum", str(rowval(row, "momentum", 0.937)),
        "--weight_decay", str(rowval(row, "weight_decay", 0.0005)),
        "--scheduler", str(rowval(row, "scheduler", "cosine")),
        "--epochs", str(rowval(row, "epochs", 50)),
        "--patience", str(rowval(row, "patience", 6)),
        "--imgsz", str(rowval(row, "imgsz", 640)),
        "--batch", str(rowval(row, "batch", 32)),
        "--num_workers", str(rowval(row, "num_workers", 16)),
        "--project", str(rowval(row, "project", DEFAULT_PROJECT)),
        "--name", str(rowval(row, "name", "yolo_run")),
        "--seed", str(rowval(row, "seed", 42)),
        "--weight_sampling", str(rowval(row, "weight_sampling", "false")),
        "--label_smoothing", str(rowval(row, "label_smoothing", 0.05)),
    ]
    return cmd

# -------------------------
# Main loop
# -------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    df = ensure_columns(df)

    for _, row in df.iterrows():
        name_val = rowval(row, "name", "yolo_run")
        run_dir = resolve_run_dir(rowval(row, "project", DEFAULT_PROJECT), name_val, PROJECT_DIR)

        # Skip already finished runs
        if job_completed(run_dir):
            now = time.time()
            log_result(row, "skipped_exists", now, now, run_dir=run_dir)
            print(f"[SKIP] {name_val} already trained.")
            continue

        start = time.time()
        try:
            cmd = yolo_cmd(row)
            print("[RUN]", " ".join(cmd))
            proc = subprocess.run(cmd, check=True, cwd=str(PROJECT_DIR))
            end = time.time()
            status = "ok" if proc.returncode == 0 else f"rc={proc.returncode}"
            log_result(row, status, start, end, run_dir=run_dir)
            print(f"[DONE] {name_val} in {(end - start)/60.0:.1f} min")

        except Exception as e:
            end = time.time()
            log_result(row, "error", start, end, err_msg=str(e), run_dir=run_dir)
            print(f"[ERROR] {name_val}: {e}")

if __name__ == "__main__":
    main()
