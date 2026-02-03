# ======================================================
# YOLO Training Script — Static & Dynamic Weighted Sampling + Aug YAML + Run Config Logging
# ======================================================
import json
import yaml
import random
import argparse
from pathlib import Path
import sys, datetime
import numpy as np
import torch
from ultralytics import YOLO
import urllib.request
from yolo_weighted_sampler import WeightedSamplerTrainer, log_class_balance

# -------------------------
# Setup
# -------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

EPOCHS_OVERRIDE = None  # set to int to force fixed epochs

# -------------------------
# Official weights
# -------------------------
MODEL_GRID = {
    "yolov11": {
        "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11s.pt"
    }
}

def ensure_model_exists(family: str, size: str) -> Path:

    model_name = f"{family}{size}.pt"
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    local_path = weights_dir / model_name

    # --- Try to use local file ---
    if local_path.exists():
        print(f"[INFO] Using local pretrained weights → {local_path}")
        return local_path

    # --- Try direct Ultralytics model registry (auto-download) ---
    print(f"[INFO] Local file not found → attempting Ultralytics download for {model_name}")
    try:
        _ = YOLO(model_name)
        cached_path = Path.home() / ".cache" / "ultralytics" / "models" / model_name
        if cached_path.exists():
            print(f"[INFO] Successfully downloaded YOLO model to cache → {cached_path}")

            urllib.request.urlretrieve(str(cached_path), str(local_path))
            print(f"[INFO] Copied cached model to → {local_path}")
            return local_path
        else:
            print("[WARN] Ultralytics cache path not found after download attempt.")
    except Exception as e:
        print(f"[WARN] Could not auto-download via Ultralytics: {e}")

    raise FileNotFoundError(
        f"[ERROR] Pretrained weights not found or could not be downloaded: {local_path}\n"
        f"Please place '{model_name}' in the 'weights/' folder manually."
    )

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_smoke_and_background(train_list_path: Path) -> tuple[int, int]:
    train_list_path = Path(train_list_path)
    if not train_list_path.exists():
        raise FileNotFoundError(f"Train list not found: {train_list_path}")
    smoke_count, bg_count = 0, 0
    with open(train_list_path, "r") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]
    for img_path in image_paths:
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if not label_path.exists():
            bg_count += 1
            continue
        lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
        if lines:
            smoke_count += 1
        else:
            bg_count += 1
    return smoke_count, bg_count

def prepare_static_balanced_yaml(base_yaml_path, balanced_list_path, output_dir):
    base_yaml = yaml.safe_load(open(base_yaml_path, "r"))
    balanced_yaml_path = Path(output_dir) / "data_balanced.yaml"
    base_yaml["train"] = str(balanced_list_path)
    yaml.safe_dump(base_yaml, open(balanced_yaml_path, "w"))
    print(f"[INFO] Created balanced data YAML → {balanced_yaml_path}")
    return balanced_yaml_path

def log_static_class_distribution(run_dir, smoke_count, bg_count):
    out_path = Path(run_dir) / "class_distribution.json"
    total = smoke_count + bg_count
    data = {
        "sampling_mode": "Static Weighted Sampling",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ratios": {
            "smoke": smoke_count,
            "background": bg_count,
            "total": total,
            "ratio_smoke": round(smoke_count / total, 4) if total else 0,
            "ratio_background": round(bg_count / total, 4) if total else 0,
        }
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"[INFO] Logged post-balancing ratios → smoke: {smoke_count}, background: {bg_count}")
    print(f"[INFO] Saved dataset image distribution → {out_path}")
    return out_path

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Train YOLO with Static/Dynamic Weighted Sampling", add_help=True)
    p.add_argument("--model_family", type=str, default="yolov11", choices=["yolov8","yolov11"])
    p.add_argument("--model_size", type=str, default="n", choices=["n","s","m","l","x"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--optimizer", type=str, default="SGD")
    p.add_argument("--lr0", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--data", type=str, default="data/pyro-sdis/splits_3fold/fold3/data_fold3.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_sampling", type=str, default="false")  # false|static|dynamic
    p.add_argument("--label_smoothing", type=float, default=0.05)
    # augmentation toggle (loads YAMLs)
    p.add_argument("--use_aug", type=str, default="False")              # "True"/"False"
    p.add_argument("--project", type=str, default="results/batch_runs/core")
    p.add_argument("--name", type=str, default="yolo_run")
    args, _ = p.parse_known_args()
    return args

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    set_seed(int(args.seed))
    epochs = int(EPOCHS_OVERRIDE) if EPOCHS_OVERRIDE else int(args.epochs)

    # run directory and config logging
    run_dir = Path(args.project) / args.name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_test").mkdir(parents=True, exist_ok=True)

    # augmentation YAML
    aug_bool = str(args.use_aug).lower() in {"1","true","t","yes","y"}
    aug_yaml = Path("src/configs/yolo_aug.yaml")
    noaug_yaml = Path("src/configs/yolo_noaug.yaml")
    cfg_path = aug_yaml if aug_bool else noaug_yaml
    print(f"[AUG] Using config: {cfg_path}")

    aug_params = {}
    if cfg_path.exists():
        try:
            aug_params = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception as e:
            print(f"[WARN] Could not parse {cfg_path}: {e}")


    ULTRA_AUG_KEYS = {
        "degrees", "translate", "scale", "shear", "perspective",
        "hsv_h", "hsv_s", "hsv_v", "flipud", "fliplr", "mosaic", "mixup",
        "copy_paste", "erasing", "mosaic_prob", "mixup_prob",
    }
    train_aug_kwargs = {k: aug_params[k] for k in ULTRA_AUG_KEYS if k in aug_params}

    # save run_config.json
    run_cfg = vars(args).copy()
    run_cfg.update({
        "epochs_effective": epochs,
        "aug_yaml_used": str(cfg_path),
        "train_aug_kwargs": train_aug_kwargs,
    })
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    # model weights
    weights_path = ensure_model_exists(args.model_family, args.model_size)

    # -------------------------
    # Sampling mode selection
    # -------------------------
    sampling_mode = str(args.weight_sampling).lower().strip()
    if sampling_mode == "false":
        print("[INFO] Weighted sampling disabled → standard YOLO.")
        data_path_to_use = args.data

    elif sampling_mode == "static":
        print("[INFO] Weighted sampling mode: Static")

        # --- Load dataset YAML and resolve original train list ---
        data_cfg = yaml.safe_load(open(args.data, "r"))
        train_list = Path(data_cfg.get("train")).resolve()
        if not train_list.exists():
            raise FileNotFoundError(f"[ERROR] Train list not found: {train_list}")

        # --- Balanced list path ---
        balanced_list = train_list.parent / "train_balanced.txt"

        # --- Deterministic oversampling setup ---
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)

        # --- Generate if missing ---
        if not balanced_list.exists():
            print(f"[INFO] No existing balanced list found → creating a new one from {train_list}")

            image_paths = [Path(p.strip()) for p in train_list.read_text().splitlines() if p.strip()]
            smoke_paths, bg_paths = [], []

            for img_path in image_paths:
                label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
                if label_path.exists() and label_path.stat().st_size > 0:
                    smoke_paths.append(str(img_path))
                else:
                    bg_paths.append(str(img_path))

            n_smoke, n_bg = len(smoke_paths), len(bg_paths)
            diff = abs(n_smoke - n_bg)

            if diff == 0:
                print("[INFO] Dataset already balanced.")
                balanced_paths = smoke_paths + bg_paths
            elif n_smoke < n_bg:
                print(f"[INFO] Oversampling smoke images by {diff}. (seed={seed})")
                smoke_paths += random.choices(smoke_paths, k=diff)
                balanced_paths = smoke_paths + bg_paths
            else:
                print(f"[INFO] Oversampling background images by {diff}. (seed={seed})")
                bg_paths += random.choices(bg_paths, k=diff)
                balanced_paths = smoke_paths + bg_paths

            random.shuffle(balanced_paths)
            balanced_list.write_text("\n".join(balanced_paths))
            print(f"[INFO] Saved new balanced train list → {balanced_list}")
        else:
            print(f"[INFO] Reusing existing balanced list → {balanced_list}")

        # --- Create balanced dataset YAML ---
        data_path_to_use = prepare_static_balanced_yaml(
            base_yaml_path=args.data,
            balanced_list_path=balanced_list,
            output_dir=run_dir
        )

        # --- Log new class ratios ---
        smoke_count, bg_count = count_smoke_and_background(balanced_list)
        log_static_class_distribution(run_dir, smoke_count, bg_count)


    elif sampling_mode == "dynamic":
        print("[INFO] Weighted sampling mode: Dynamic (WeightedRandomSampler)")
        data_path_to_use = args.data

    else:
        raise ValueError(f"Invalid weight_sampling mode: {sampling_mode}")

    # -------------------------
    # Train
    # -------------------------
    print(f"[INFO] Training {args.model_family}{args.model_size} | epochs={epochs} | aug={aug_bool} | mode={sampling_mode}")

    if sampling_mode == "dynamic":
        trainer = WeightedSamplerTrainer(overrides=dict(
            data=str(data_path_to_use),
            model=str(weights_path),
            project=str(args.project),
            name=str(args.name),
            exist_ok=True,
            save_json=True,
            save_dir=str(run_dir),
            epochs=epochs,
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            optimizer=str(args.optimizer),
            lr0=float(args.lr0),
            weight_decay=float(args.weight_decay),
            momentum=float(args.momentum),
            patience=int(args.patience),
            save_period=2,
            seed=int(args.seed),
            deterministic=True,
            verbose=True,
            amp=True,
            cos_lr=(args.scheduler == "cosine"),
            workers=int(args.num_workers),
            label_smoothing=float(args.label_smoothing),

            **train_aug_kwargs,
        ))
        trainer.save_dir = run_dir
        trainer.set_callback("on_train_epoch_end", log_class_balance)

        results = trainer.train()

        # expose model handle for param count below
        model_for_params = trainer.model

    else:
        # standard YOLO path
        model = YOLO(str(weights_path))
        results = model.train(
            data=str(data_path_to_use),
            project=str(args.project),
            name=str(args.name),
            exist_ok=True,
            save_json=True,
            epochs=epochs,
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            optimizer=str(args.optimizer),
            lr0=float(args.lr0),
            weight_decay=float(args.weight_decay),
            momentum=float(args.momentum),
            patience=int(args.patience),
            save_period=2,
            seed=int(args.seed),
            deterministic=True,
            verbose=True,
            amp=True,
            cos_lr=(args.scheduler == "cosine"),
            workers=int(args.num_workers),
            label_smoothing=float(args.label_smoothing),

            **train_aug_kwargs,
        )
        model_for_params = model.model

    # -------------------------
    # Metadata
    # -------------------------
    (run_dir / "hardware.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }, indent=2))

    try:
        params = sum(p.numel() for p in model_for_params.parameters())
    except Exception:
        params = "N/A"
    (run_dir / "model_card.txt").write_text(
        f"Model: {args.model_family}{args.model_size}\nParameters: {params}\n"
    )
    print(f"[DONE] Finished → {run_dir}")

if __name__ == "__main__":
    main()
