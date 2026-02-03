"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions in the CONFIG section below
(e.g., IMG_ROOT, COCO_JSON) to match your local directory structure.
"""

from __future__ import annotations
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from training.train_faster_rcnn import get_fasterrcnn_model

# ========
# CONFIG
# ========

# --- Full FIgLib COCO json listing all images (weak-label set) ---
COCO_JSON = Path("/path/to/annotations/json")

# --- Root directory for images ---
IMG_ROOT = Path("/path/to/images")

# Image-level GT CSV derived from filename encoding (+ / -)
GT_CSV = Path(r"path\to\csv")

# --- Output folders ---
OUT_ROOT = Path(r"output\dir")
CACHE_ROOT_YOLO = OUT_ROOT / "yolo_cache" / "figlib_full"
CACHE_ROOT_RCNN = OUT_ROOT / "rcnn_cache" / "figlib_full"
METRICS_OUT = OUT_ROOT / "metrics"
PER_IMAGE_OUT = OUT_ROOT / "per_image"

EXCLUDE_TOKENS = {"bbox", "transition"}

# --- Inference params ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_YOLO = 64
BATCH_RCNN = 8

RCNN_STORE_FULL_DETS = True

RCNN_FP16 = True
RCNN_EMPTY_CACHE_EVERY = 50


YOLO_CONF = 0.001
YOLO_IOU = 0.70


RCNN_SCORE_THRESH: Optional[float] = None

CONF_THRESH = 0.30

SKIP_IF_CACHE_EXISTS = True


# --- Winners (EDIT PATHS) ---
# Each tuple: (config_name, checkpoint_path)
YOLO_WINNERS: List[Tuple[str, Path]] = [
    ("yolov11s_aug_nows_0.001_seed_22", Path(r"path\to\checkpoint")),
]
RCNN_WINNERS: List[Tuple[str, Path]] = [
    ("rcnn_aug_nows_lr00025_fold3_seed_500", Path(r"path\to\checkpoint")),
]

# ===========
# HELPERS
# ===========

def norm_rel(s: str) -> str:

    return str(s).replace("\\", "/").lstrip("./")

def is_excluded(rel_path: str) -> bool:
    if not EXCLUDE_TOKENS:
        return False
    parts = set(norm_rel(rel_path).split("/"))
    return any(tok in parts for tok in EXCLUDE_TOKENS)

def load_coco_images(coco_json: Path) -> List[Dict[str, Any]]:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    if not images:
        raise ValueError("COCO JSON has no 'images' entries.")
    return images

def save_pkl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy
    }

def build_items() -> List[Dict[str, Any]]:

    images = load_coco_images(COCO_JSON)

    items: List[Dict[str, Any]] = []
    for im in images:
        rel = norm_rel(im["file_name"])
        if is_excluded(rel):
            continue

        abs_path = (IMG_ROOT / Path(rel)).resolve()

        items.append({
            "image_id": int(im["id"]),
            "image_rel": rel,
            "event": im.get("event", "") or (Path(rel).parts[0] if Path(rel).parts else ""),
            "image_abs": str(abs_path),
        })

    if not items:
        raise RuntimeError("No images remain after exclusions. Check EXCLUDE_TOKENS and COCO file_name paths.")
    return items

def load_gt_mapping(gt_csv: Path) -> Dict[str, int]:
    """
    GT CSV must include columns:
      - image_rel
      - gt_smoke (0/1)
    """
    df = pd.read_csv(gt_csv)
    if "image_rel" not in df.columns or "gt_smoke" not in df.columns:
        raise ValueError(f"GT CSV missing required columns. Found: {df.columns.tolist()}")

    df["image_rel"] = df["image_rel"].astype(str).str.replace("\\", "/", regex=False).str.lstrip("./")
    df["gt_smoke"] = df["gt_smoke"].astype(int)

    n_pos = int(df["gt_smoke"].sum())
    n_total = int(len(df))
    print(f"[INFO] Loaded GT CSV: {gt_csv} | positives={n_pos}/{n_total} ({n_pos/n_total:.3f})")

    return dict(zip(df["image_rel"], df["gt_smoke"]))


def _rcnn_autocast_context(device: str):
    if (device == "cuda") and RCNN_FP16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    # no-op context
    from contextlib import nullcontext
    return nullcontext()


def _load_image_tensor(path: str, tf: T.ToTensor) -> torch.Tensor:
    # PIL load on CPU, tensor on CPU
    img = Image.open(path).convert("RGB")
    return tf(img)


# ========================
# Faster R-CNN loader
# ========================

@dataclass
class EvalConfig:
    pretrained: bool = False
    num_classes: int = 2
    label_smoothing: float = 0.0

def load_frcnn(ckpt_path: Path, device: str):
    cfg = EvalConfig()
    model = get_fasterrcnn_model(cfg)

    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    if RCNN_SCORE_THRESH is not None and hasattr(model, "roi_heads"):
        model.roi_heads.score_thresh = float(RCNN_SCORE_THRESH)

    return model

# ====================
# INFERENCE + CACHE
# ====================

@torch.inference_mode()
def cache_yolo(config_name: str, ckpt: Path, items: List[Dict[str, Any]]) -> Path:
    out_pkl = (CACHE_ROOT_YOLO / config_name / "predictions.pkl")
    if SKIP_IF_CACHE_EXISTS and out_pkl.exists():
        print(f"YOLO cache exists, skipping inference: {config_name}")
        return out_pkl
    if not ckpt.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt}")

    print(f"\nYOLO | {config_name}")
    yolo = YOLO(str(ckpt))

    preds: List[Dict[str, Any]] = []
    abs_paths = [it["image_abs"] for it in items]

    n_batches = (len(abs_paths) + BATCH_YOLO - 1) // BATCH_YOLO
    pbar = tqdm(range(0, len(abs_paths), BATCH_YOLO),
                total=n_batches, desc=f"YOLO infer ({config_name})", unit="batch", leave=False)

    for i in pbar:
        batch_paths = abs_paths[i:i + BATCH_YOLO]
        batch_items = items[i:i + BATCH_YOLO]

        outs = yolo.predict(batch_paths, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)

        for it, abs_path, out in zip(batch_items, batch_paths, outs):
            if out.boxes is None:
                boxes = np.zeros((0, 4), np.float32)
                scores = np.zeros((0,), np.float32)
                labels = np.zeros((0,), np.int64)
            else:
                boxes = out.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = out.boxes.conf.cpu().numpy().astype(np.float32)
                labels = (out.boxes.cls.long().cpu().numpy().astype(np.int64) + 1)

            preds.append({
                "image_id": it["image_id"],
                "image_rel": it["image_rel"],
                "image_path": abs_path,
                "event_id": it["event"],
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

    save_pkl(out_pkl, preds)
    del yolo
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved YOLO predictions → {out_pkl}")
    return out_pkl


@torch.inference_mode()
def cache_rcnn(config_name: str, ckpt: Path, items: List[Dict[str, Any]]) -> Path:

    out_pkl = (CACHE_ROOT_RCNN / config_name / "predictions.pkl")
    if SKIP_IF_CACHE_EXISTS and out_pkl.exists():
        print(f"RCNN cache exists, skipping inference: {config_name}")
        return out_pkl
    if not ckpt.exists():
        raise FileNotFoundError(f"RCNN checkpoint not found: {ckpt}")

    print(f"\nRCNN | {config_name} (OOM-safe)")
    model = load_frcnn(ckpt, DEVICE)

    if (RCNN_SCORE_THRESH is not None) and hasattr(model, "roi_heads"):
        model.roi_heads.score_thresh = float(RCNN_SCORE_THRESH)

    tf = T.ToTensor()
    preds: List[Dict[str, Any]] = []

    n_batches = (len(items) + BATCH_RCNN - 1) // BATCH_RCNN
    pbar = tqdm(
        range(0, len(items), BATCH_RCNN),
        total=n_batches,
        desc=f"RCNN infer ({config_name})",
        unit="batch",
        leave=False
    )

    autocast_ctx = _rcnn_autocast_context(DEVICE)

    for bi, start in enumerate(pbar):
        batch = items[start:start + BATCH_RCNN]

        imgs_cpu = [_load_image_tensor(it["image_abs"], tf) for it in batch]
        imgs = [im.to(DEVICE, non_blocking=True) for im in imgs_cpu]

        with autocast_ctx:
            outputs = model(imgs)

        for it, out in zip(batch, outputs):
            scores = out["scores"].detach().float().cpu().numpy().astype(np.float32)

            if RCNN_STORE_FULL_DETS:
                boxes = out["boxes"].detach().float().cpu().numpy().astype(np.float32)
                labels = out["labels"].detach().cpu().numpy().astype(np.int64)
            else:

                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

            preds.append({
                "image_id": it["image_id"],
                "image_rel": it["image_rel"],
                "image_path": it["image_abs"],
                "event_id": it["event"],
                "scores": scores,
                "boxes": boxes,
                "labels": labels,
            })

        del imgs, imgs_cpu, outputs, batch
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            if RCNN_EMPTY_CACHE_EVERY and ((bi + 1) % RCNN_EMPTY_CACHE_EVERY == 0):
                torch.cuda.empty_cache()

    save_pkl(out_pkl, preds)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved RCNN predictions → {out_pkl}")
    return out_pkl

# =============
# EVALUATION
# =============

def evaluate_cache(cache_pkl: Path,
                   arch: str,
                   config_name: str,
                   items: List[Dict[str, Any]],
                   gt_by_rel: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    preds = load_pkl(cache_pkl)
    pred_by_rel: Dict[str, Dict[str, Any]] = {p["image_rel"]: p for p in preds}

    rows = []
    pbar = tqdm(items, desc=f"Eval ({arch}, {config_name})", unit="img", leave=False)
    for it in pbar:
        rel = norm_rel(it["image_rel"])
        gt = int(gt_by_rel.get(rel, 0))

        p = pred_by_rel.get(rel)
        if p is None:
            scores = np.zeros((0,), np.float32)
            num_boxes = 0
        else:
            scores = np.asarray(p.get("scores", []), dtype=np.float32)
            num_boxes = int(scores.size)

        pred_smoke = int(np.any(scores >= CONF_THRESH))

        rows.append({
            "architecture": arch,
            "model": config_name,
            "conf_thresh": CONF_THRESH,
            "image_rel": rel,
            "event": it["event"],
            "gt_smoke": gt,
            "pred_smoke": pred_smoke,
            "num_boxes": num_boxes,
            "max_conf": float(scores.max()) if num_boxes > 0 else 0.0,
            "mean_conf": float(scores.mean()) if num_boxes > 0 else 0.0,
        })

    df = pd.DataFrame(rows)

    y_true = df["gt_smoke"].to_numpy(dtype=np.int32)
    y_pred = df["pred_smoke"].to_numpy(dtype=np.int32)
    metrics = compute_binary_metrics(y_true, y_pred)

    n_pos = int(df["gt_smoke"].sum())
    n_neg = int((df["gt_smoke"] == 0).sum())

    summary = pd.DataFrame([{
        "architecture": arch,
        "model": config_name,
        "conf_thresh": CONF_THRESH,
        "n_images": len(df),
        "n_pos_gt": n_pos,
        "n_neg_gt": n_neg,
        **metrics,
        "mean_num_boxes": float(df["num_boxes"].mean()),
        "mean_max_conf": float(df["max_conf"].mean()),
        "mean_mean_conf": float(df["mean_conf"].mean()),
    }])

    return df, summary

# ========
# MAIN
# ========

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT_YOLO.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT_RCNN.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.mkdir(parents=True, exist_ok=True)
    PER_IMAGE_OUT.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")
    items = build_items()
    print(f"[INFO] Eval set size (after exclusions): {len(items)} images")

    gt_by_rel = load_gt_mapping(GT_CSV)

    item_rels = {norm_rel(x["image_rel"]) for x in items}
    gt_rels = set(gt_by_rel.keys())
    missing = sorted(item_rels - gt_rels)[:10]
    if missing:
        raise RuntimeError(
            f"[ERROR] {len(item_rels - gt_rels)} images missing in GT CSV. "
            f"Examples: {missing}"
        )
    print("[INFO] GT CSV covers all evaluation images.")

    # Build config list
    all_configs: List[Tuple[str, str, Path]] = []
    all_configs += [("YOLOv11s", name, ckpt) for name, ckpt in YOLO_WINNERS]
    all_configs += [("Faster R-CNN", name, ckpt) for name, ckpt in RCNN_WINNERS]

    if not all_configs:
        raise RuntimeError("No configs provided.")

    print(f"[INFO] Running {len(all_configs)} configs with CONF_THRESH={CONF_THRESH}")

    all_summaries: List[pd.DataFrame] = []

    cfg_pbar = tqdm(all_configs, desc="All configs", unit="cfg")
    for arch, name, ckpt in cfg_pbar:
        cfg_pbar.set_postfix_str(name)

        if arch == "YOLOv11s":
            cache_pkl = cache_yolo(name, ckpt, items)
        else:
            cache_pkl = cache_rcnn(name, ckpt, items)

        per_img_df, summary_df = evaluate_cache(cache_pkl, arch, name, items, gt_by_rel)

        per_img_path = PER_IMAGE_OUT / f"{name}_per_image.csv"
        metrics_path = METRICS_OUT / f"{name}_metrics.csv"
        per_img_df.to_csv(per_img_path, index=False)
        summary_df.to_csv(metrics_path, index=False)
        all_summaries.append(summary_df)

        print(f"Saved per-image → {per_img_path}")
        print(f"Saved metrics   → {metrics_path}")

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_out = OUT_ROOT / "figlib_full_imagelevel_metrics_all_models.csv"
    combined.to_csv(combined_out, index=False)
    print(f"\nSaved combined metrics table → {combined_out}")
    print("\nDone.")

if __name__ == "__main__":
    main()
