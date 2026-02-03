from __future__ import annotations

import csv
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator

from training.rcnn_utils import (
    visualize_augmentations_before_after,
    log_class_balance_rcnn,
    plot_training_curves,
)

# -------------------------
# Helpers
# -------------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set global random seeds for reproducible training.

    Args:
        seed: Random seed for Python, NumPy, and PyTorch.
        deterministic: Whether to force deterministic CuDNN behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_tqdm(*args, **kwargs):
    """
    Provide a tqdm progress bar constructor.

    Returns:
        tqdm type to construct progress bars in console mode.
    """
    from tqdm import tqdm as _tqdm
    return _tqdm


def collate_fn(batch):
    """
    Default collate function for detection tasks.

    Returns:
        Tuple[List[Tensor], List[Dict]] where images is a list of
        CxHxW tensors and targets is a list of detection dicts.
    """
    return tuple(zip(*batch))


def ensure_outdir(path: Path) -> None:
    """
    Create an output directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def log_epoch_stats(history: List[Dict], out_dir: Path) -> None:
    """
    Append epoch-level train/val statistics to a CSV log.

    Merges with existing training_log.csv, overwriting rows for epochs
    that already exist so resumed runs remain consistent.

    Args:
        history: List of epoch dictionaries to log.
        out_dir: Output directory where training_log.csv is stored.
    """
    ensure_outdir(out_dir)
    log_csv = out_dir / "training_log.csv"

    preferred_order = [
        "epoch", "lr", "val_mode",
        "train_loss_classifier", "train_loss_box_reg",
        "train_loss_objectness", "train_loss_rpn_box_reg",
        "val_loss_classifier", "val_loss_box_reg",
        "val_loss_objectness", "val_loss_rpn_box_reg",
        "mAP_50", "mAP_50_95", "mAP_75",
        "mAP_small", "mAP_medium", "mAP_large",
        "AR_1", "AR_10", "AR_100",
        "AR_small", "AR_medium", "AR_large",
        "avg_dets_per_img",
        "train_peak_gpu_mem", "val_peak_gpu_mem",
        "loss_type", "time_sec",
    ]

    all_keys = set()
    for row in history:
        all_keys.update(row.keys())

    fieldnames = [k for k in preferred_order if k in all_keys] + [
        k for k in sorted(all_keys) if k not in preferred_order
    ]

    norm_history = []
    for row in history:
        norm_row = {k: (row[k] if k in row else float("nan")) for k in fieldnames}
        norm_history.append(norm_row)

    if log_csv.exists():
        with open(log_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            old_rows = [r for r in reader]

        old_epochs = {
            int(r["epoch"]): r
            for r in old_rows
            if "epoch" in r and r["epoch"] != ""
        }

        for row in norm_history:
            ep = int(row["epoch"])
            old_epochs[ep] = row

        merged = [old_epochs[ep] for ep in sorted(old_epochs.keys())]

        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)

        print(f"Updated training log at {log_csv} ({len(merged)} epochs, merged without duplicates)")
    else:
        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(norm_history)
        print(f"Created training log at {log_csv} ({len(norm_history)} epochs)")

# -------------------------
# Config and scheduler
# -------------------------

@dataclass
class TrainConfig:
    """
    Configuration container for Faster R-CNN training.

    Stores dataset paths, optimization hyperparameters, scheduler
    settings, augmentation flags, and evaluation options.
    """
    images_root_train: Path
    images_root_val: Path
    images_root_test: Path
    train_json: Path
    val_json: Path
    test_json: Path
    out_dir: Path

    epochs: int = 20
    batch_size: int = 4
    num_workers: int = 8
    amp: bool = True
    patience: int = 6
    max_batches: Optional[int] = None
    seed: int = 42
    deterministic: bool = True
    empty_cache: bool = True
    resume_from: str = "last"  # "last", "best", "none"

    lr_scheduler_type: str = "cosine"
    learning_rate: float = 5e-3
    lr_milestones: Tuple[int, int] = (8, 15)
    lrf: float = 0.01
    lr_factor: float = 0.7
    lr_plateau_patience: int = 2
    lr_plateau_threshold: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 1e-4

    pretrained: bool = True
    num_classes: int = 2

    eval_mode: str = "coco"
    eval_every_n: int = 2

    loss_type: str = "ce"
    samples_per_class: Optional[List[int]] = None

    use_augmentations: bool = True
    weight_sampling: str = "false"  # "false" | "static" | "dynamic"
    label_smoothing: float = 0.1
    warmup_epochs: int = 3

    eval_score_thresh: float = 0.25
    eval_nms_iou: float = 0.45
    eval_max_dets: int = 100

    dataset_name: str = "pyro-sdis-full"
    class_names: Dict[int, str] = field(default_factory=lambda: {1: "smoke"})


class WarmupPlateauScheduler:
    """
    LR scheduler: warmup followed by ReduceLROnPlateau.

    Provides a unified step() interface so it can be checkpointed and
    restored like built-in schedulers.
    """
    def __init__(self, optimizer, warmup_scheduler, plateau_scheduler, warmup_epochs: int):
        """
        Initialize the composed scheduler.

        Args:
            optimizer: Optimizer whose LR is being scheduled.
            warmup_scheduler: LambdaLR (or similar) for warmup phase.
            plateau_scheduler: ReduceLROnPlateau for later epochs.
            warmup_epochs: Number of warmup epochs.
        """
        self.optimizer = optimizer
        self.warmup = warmup_scheduler
        self.plateau = plateau_scheduler
        self.warmup_epochs = warmup_epochs
        self.last_epoch = -1
        self._in_warmup = True
        self._last_lr = optimizer.param_groups[0]["lr"]

    def step(self, metric=None, epoch=None):
        """
        Advance the scheduler by one epoch.

        Uses the warmup scheduler for the first warmup_epochs, then
        switches to ReduceLROnPlateau, using the provided metric.
        """
        self.last_epoch += 1

        if self._in_warmup and self.last_epoch < self.warmup_epochs:
            self.warmup.step()
            return

        if self._in_warmup:
            print(
                f"[Scheduler] Warm-up finished at epoch {self.warmup_epochs}. "
                f"Switching to ReduceLROnPlateau."
            )
            self._in_warmup = False
            self.plateau.last_epoch = -1

        prev_lr = self.optimizer.param_groups[0]["lr"]

        if metric is not None:
            self.plateau.step(metric)
        else:
            self.plateau.step(0.0)

        new_lr = self.optimizer.param_groups[0]["lr"]
        state = self.plateau.state_dict()
        print(
            f"[Plateau Debug] metric={metric:.4f}, best={state['best']:.4f}, "
            f"bad_epochs={state['num_bad_epochs']}, lr={new_lr:.6f}"
        )

        if new_lr < prev_lr:
            print(
                f"Epoch {self.last_epoch}: ReduceLROnPlateau reducing learning rate "
                f"from {prev_lr:.6f} to {new_lr:.6f}"
            )

    def state_dict(self):
        """
        Return a merged state dict so checkpointing works.

        Returns:
            Dictionary containing warmup and plateau scheduler states
            and the last epoch index.
        """
        return {
            "warmup": self.warmup.state_dict(),
            "plateau": self.plateau.state_dict(),
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state):
        """
        Restore internal scheduler state from a checkpoint dict.

        Args:
            state: Dict returned by state_dict().
        """
        self.warmup.load_state_dict(state["warmup"])
        self.plateau.load_state_dict(state["plateau"])
        self.last_epoch = state.get("last_epoch", -1)


# -------------------------
# Reproducibility utilities
# -------------------------

def save_env(cfg: TrainConfig, out_dir: Path) -> None:
    """
    Save configuration and environment metadata to disk.

    Writes:
      - config.json: TrainConfig fields
      - env_info.json: Python, platform, PyTorch, CUDA, GPU, and Git commit

    Args:
        cfg: Training configuration.
        out_dir: Output directory for metadata files.
    """
    ensure_outdir(out_dir)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    env_info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        env_info["git_commit"] = commit
    except Exception:
        env_info["git_commit"] = "N/A"

    with open(out_dir / "env_info.json", "w") as f:
        json.dump(env_info, f, indent=2)

    print(f"Saved environment info to {out_dir}")


def save_dataset_stats(
    dls: Dict[str, DataLoader],
    out_dir: Path,
    dataset_name: str = "unknown",
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict]:
    """
    Compute and save dataset statistics per split.

    Iterates over train, val, and test loaders to count images, boxes,
    and per-class image/box counts. Stores the result in dataset_stats.json.

    Args:
        dls: Mapping of split name to DataLoader.
        out_dir: Output directory for dataset_stats.json.
        dataset_name: Name of the dataset for logging.
        class_names: Optional mapping from class id to human-readable name.

    Returns:
        Dictionary of aggregated statistics keyed by split.
    """
    stats = {"dataset": dataset_name}

    for split, dl in dls.items():
        img_count, box_count = 0, 0
        class_counter = Counter()
        smoke_images = 0
        background_images = 0

        pbar = tqdm(dl, desc=f"Collecting stats [{split}]", unit="batch")
        for images, targets in pbar:
            img_count += len(images)
            for t in targets:
                n = t["boxes"].shape[0]
                box_count += n
                if n == 0:
                    background_images += 1
                else:
                    smoke_images += 1
                    if "labels" in t:
                        class_counter.update(t["labels"].cpu().numpy().tolist())

        per_class_boxes = {}
        for cls_id, count in class_counter.items():
            if class_names and cls_id in class_names:
                per_class_boxes[class_names[cls_id]] = count
            else:
                per_class_boxes[str(cls_id)] = count

        per_class_images = {
            "smoke": smoke_images,
            "background": background_images,
        }

        stats[split] = {
            "images": img_count,
            "boxes": box_count,
            "per_class_boxes": per_class_boxes,
            "per_class_images": per_class_images,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_file = out_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved dataset stats → {stats_file}")
    return stats


def generate_train_balanced_json(cfg: TrainConfig, dst_path: Optional[Path] = None) -> Path:
    """
    Create a balanced COCO train split by oversampling background images.

    Args:
        cfg: Training configuration with train_json set.
        dst_path: Optional destination path for the balanced JSON.

    Returns:
        Path to the balanced COCO JSON file.
    """
    src_json = Path(cfg.train_json)

    if dst_path is None:
        dst_path = src_json.parent / "train_balanced.json"

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    seed = int(getattr(cfg, "seed", 42))
    random.seed(seed)

    coco = COCO(str(src_json))
    img_ids = coco.getImgIds()

    smoke_img_ids, back_img_ids = [], []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        (back_img_ids if len(ann_ids) == 0 else smoke_img_ids).append(img_id)

    n_smoke, n_back = len(smoke_img_ids), len(back_img_ids)
    print(f"[INFO] Before balancing → smoke: {n_smoke}, background: {n_back}")

    if n_back >= n_smoke or n_back == 0:
        print(
            "[INFO] Background count already ≥ smoke (or no background present) "
            "→ no oversampling."
        )
        with open(src_json, "r") as f:
            data = json.load(f)
        with open(dst_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved balanced file → {dst_path}")
        return dst_path

    num_to_add = n_smoke - n_back
    print(f"[INFO] Oversampling background by +{num_to_add} images (to match smoke={n_smoke}).")

    with open(src_json, "r") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    max_img_id = max((img["id"] for img in data["images"]), default=0)

    balanced_images = list(data["images"])
    balanced_annotations = list(data["annotations"])

    for _ in range(num_to_add):
        orig_id = random.choice(back_img_ids)
        dup_img = dict(images_by_id[orig_id])
        max_img_id += 1
        dup_img["id"] = max_img_id
        dup_img["duplicate_of"] = orig_id
        balanced_images.append(dup_img)

    balanced_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": balanced_images,
        "annotations": balanced_annotations,
        "categories": data.get("categories", []),
    }

    with open(dst_path, "w") as f:
        json.dump(balanced_data, f, indent=2)

    print(f"[INFO] After balancing  → smoke: {n_smoke}, background: {n_back + num_to_add}")
    print(f"[INFO] Saved balanced COCO → {dst_path}")
    return dst_path


def build_per_image_binary_labels(dataset: "CocoDetDataset") -> torch.Tensor:
    """
    Build binary per-image labels for a COCO detection dataset.
    Labels are 1 for images with at least one annotation and 0 otherwise.

    Args:
        dataset: CocoDetDataset instance.

    Returns:
        Tensor of shape [N] with 0/1 labels per image.
    """
    labels = []
    coco = dataset.coco
    for img_id in dataset.ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        labels.append(1 if len(anns) > 0 else 0)
    return torch.as_tensor(labels, dtype=torch.long)


def compute_blended_class_weights(counts: torch.Tensor, smoke_target: float = 0.5) -> torch.Tensor:
    """
    Compute class weights blending inverse frequency and target mix.

    Args:
        counts: Tensor with counts for [background, smoke].
        smoke_target: Target fraction of smoke samples per epoch.

    Returns:
        Tensor of length 2 with weights for [background, smoke].
    """
    total = counts.sum().clamp_min(1)
    if (counts == 0).any():
        return torch.ones(2, dtype=torch.float32)

    eps = 1e-8
    gamma = 0.3
    freq = counts.float() / total
    w_temp = (1.0 / freq.clamp_min(eps)).pow(gamma)
    w_temp = w_temp / w_temp.mean()

    t = float(smoke_target)
    w_bg = (1.0 - t) / counts[0].clamp_min(1).float()
    w_sm = t / counts[1].clamp_min(1).float()
    w_tgt = torch.tensor([w_bg, w_sm], dtype=torch.float32)
    w_tgt = w_tgt / w_tgt.mean()

    alpha = 0.5
    w = (1 - alpha) * w_temp + alpha * w_tgt
    w = w.clamp(0.7, 1.6)
    w = w / w.mean()

    N_bg = max(int(counts[0]), 1)
    N_sm = max(int(counts[1]), 1)
    r_bg_target = 1.0 - t
    s_bg = (
        (r_bg_target / max(1.0 - r_bg_target, 1e-12))
        * (float(w[1]) * N_sm)
        / max(float(w[0]) * N_bg, 1e-12)
    )
    w = w.clone()
    w[0] = w[0] * float(s_bg)
    w = w / w.mean()
    return w


def make_train_loader_with_sampler(
    dataset: "CocoDetDataset",
    batch_size: int,
    num_workers: int,
    mode: str,
    out_dir: Path,
) -> DataLoader:
    """
    Build a training DataLoader with WeightedRandomSampler.

    Computes binary image labels and class weights to approximate a
    balanced (50/50) mix of background and smoke, logs the planned
    distribution, and returns a DataLoader with replacement sampling.

    Args:
        dataset: CocoDetDataset for training.
        batch_size: Batch size for the loader.
        num_workers: Number of worker processes.
        mode: Label for logging (e.g. "dynamic").
        out_dir: Output directory for class balance logs.

    Returns:
        DataLoader with a WeightedRandomSampler attached.
    """
    image_labels = build_per_image_binary_labels(dataset)
    counts = torch.bincount(image_labels, minlength=2)
    total = counts.sum().item()

    class_w = compute_blended_class_weights(counts, smoke_target=0.5)
    weights = class_w.index_select(0, image_labels).cpu()

    raw_bg_ratio = counts[0].item() / total
    raw_sm_ratio = counts[1].item() / total

    balanced_total = len(dataset)
    balanced_bg = int(balanced_total * 0.5)
    balanced_sm = balanced_total - balanced_bg
    balanced_bg_ratio = balanced_bg / balanced_total
    balanced_sm_ratio = balanced_sm / balanced_total

    print(f"[INFO] WeightedRandomSampler enabled — {mode}.")
    print(f"[INFO] Dataset counts: {counts.tolist()} → Weights: {class_w.tolist()}")
    print(
        f"[INFO] Raw composition → Background: {raw_bg_ratio:.3f} | "
        f"Smoke: {raw_sm_ratio:.3f}"
    )
    print(
        f"[INFO] Balanced (expected) → Background: {balanced_bg_ratio:.3f} | "
        f"Smoke: {balanced_sm_ratio:.3f}"
    )

    log_path = out_dir / "class_balance_log.json"
    record = {
        "epoch": None,
        "raw_smoke": int(counts[1].item()),
        "raw_background": int(counts[0].item()),
        "raw_ratio_smoke": round(float(raw_sm_ratio), 4),
        "raw_ratio_background": round(float(raw_bg_ratio), 4),
        "raw_total": int(total),
        "balanced_smoke": balanced_sm,
        "balanced_background": balanced_bg,
        "balanced_ratio_smoke": round(float(balanced_sm_ratio), 4),
        "balanced_ratio_background": round(float(balanced_bg_ratio), 4),
        "balanced_total": int(balanced_total),
    }

    try:
        if log_path.exists():
            existing = json.loads(log_path.read_text())
            if isinstance(existing, list):
                existing = {
                    "run_name": out_dir.name,
                    "project": str(out_dir.parent),
                    "records": existing,
                }
            existing.setdefault("records", []).append(record)
        else:
            existing = {
                "run_name": out_dir.name,
                "project": str(out_dir.parent),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epochs_total": None,
                "batch_size": int(batch_size),
                "sampling_mode": f"{mode.title()} WeightedRandomSampler",
                "records": [record],
            }

        log_path.write_text(json.dumps(existing, indent=2))
        print(
            f"[INFO] Logged class balance → raw_total={total}, "
            f"balanced_total={balanced_total}"
        )
    except Exception as e:
        print(f"[WARN] Could not write class_balance_log.json: {e}")

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader

# -------------------------
# Dataset
# -------------------------

class CocoDetDataset(torch.utils.data.Dataset):
    """
    COCO-format detection dataset compatible with Albumentations.

    Handles:
      - Bounding box conversion from [x, y, w, h] to [x1, y1, x2, y2]
      - Empty image annotations
      - Albumentations transforms with bbox_params
    """

    def __init__(self, images_dir: Path, ann_json: Path, transforms=None):
        """
        Initialize the dataset.

        Args:
            images_dir: Directory containing image files.
            ann_json: Path to COCO annotations JSON file.
            transforms: Optional Albumentations Compose pipeline.
        """
        self.images_dir = Path(images_dir)
        self.ann_json = str(ann_json)
        self.coco = COCO(self.ann_json)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.ids)

    def __getitem__(self, index: int):
        """
        Load and return a single image and its target dict.

        Returns:
            Tuple (image, target) where image is an RGB tensor after
            transforms and target is a dict with keys:
            boxes, labels, image_id, area, iscrowd.
        """
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = self.images_dir / img_info["file_name"]

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"[ERROR] Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if self.transforms is not None:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                labels=labels,
            )
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([img_id], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([img_id], dtype=torch.int64),
                "area": area,
                "iscrowd": iscrowd,
            }

        return img, target

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        """
        Convert numpy and torch types into JSON-friendly Python types.

        Args:
            obj: Arbitrary object.

        Returns:
            JSON-serializable equivalent.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: CocoDetDataset._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CocoDetDataset._to_jsonable(v) for v in obj]
        return obj


def make_transforms(is_train: bool, use_aug: bool, out_dir: Path):
    """
    Build Albumentations transforms for Faster R-CNN.

    For training, returns a richer augmentation pipeline when
    use_aug is True; for validation/test returns a minimal pipeline.

    Args:
        is_train: Whether the transforms are for training.
        use_aug: Whether to include heavy augmentations.
        out_dir: Output directory (for optional visualization hooks).

    Returns:
        Albumentations Compose object.
    """
    if is_train and use_aug:
        tfms = [
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.08, 0.08),
                rotate=(-8, 8),
                shear=(-4, 4),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(128, 128, 128),
                p=0.6,
            ),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=6,
                sat_shift_limit=16,
                val_shift_limit=10,
                p=0.35,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.3,
                p=0.35,
            ),
            A.OneOf(
                [
                    A.RandomFog(fog_coef_range=(0.05, 0.15), alpha_coef=0.03),
                    A.RandomRain(
                        slant_range=(-5, 5),
                        blur_value=2,
                        brightness_coefficient=0.95,
                    ),
                    A.MotionBlur(blur_limit=(3, 7)),
                ],
                p=0.25,
            ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    else:
        tfms = [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]

    transform = A.Compose(
        tfms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
        ),
    )
    return transform


def make_dataloaders(cfg: TrainConfig) -> Dict[str, DataLoader]:
    """
    Construct train, validation, and test DataLoaders.

    Applies optional static or dynamic class balancing for the train
    loader and uses consistent transforms across splits.

    Args:
        cfg: Training configuration.

    Returns:
        Dictionary mapping "train", "val", "test" to DataLoaders.
    """
    train_tf = make_transforms(True, cfg.use_augmentations, cfg.out_dir)
    val_tf = make_transforms(False, False, cfg.out_dir)

    ws = str(cfg.weight_sampling).lower().strip()

    if ws == "static":
        bal_path = Path(cfg.train_json).parent / "train_balanced.json"
        if not bal_path.exists():
            print("[INFO] No train_balanced.json found → generating new one")
            bal_path = generate_train_balanced_json(cfg, dst_path=bal_path)
        train_ds = CocoDetDataset(cfg.images_root_train, bal_path, transforms=train_tf)
    else:
        train_ds = CocoDetDataset(cfg.images_root_train, cfg.train_json, transforms=train_tf)

    val_ds = CocoDetDataset(cfg.images_root_val, cfg.val_json, transforms=val_tf)
    test_ds = CocoDetDataset(cfg.images_root_test, cfg.test_json, transforms=val_tf)

    if ws in ["false", "static"]:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    elif ws == "dynamic":
        train_loader = make_train_loader_with_sampler(
            train_ds,
            cfg.batch_size,
            cfg.num_workers,
            mode="dynamic",
            out_dir=cfg.out_dir,
        )
    else:
        raise ValueError(f"Invalid weight_sampling: {cfg.weight_sampling}")

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

# -------------------------
# Model
# -------------------------

def get_fasterrcnn_model(cfg: TrainConfig) -> nn.Module:
    """
    Build a Faster R-CNN detector tailored for smoke detection.

    Adjustments include:
      - Anchor sizes for small objects
      - Relaxed RPN IoU thresholds
      - Increased proposal counts
      - Custom ROI classifier with label smoothing

    Args:
        cfg: Training configuration.

    Returns:
        Configured Faster R-CNN model.
    """
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if cfg.pretrained else None,
        weights_backbone=None,
    )

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model.rpn.anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    model.rpn.fg_iou_thresh = 0.4
    model.rpn.bg_iou_thresh = 0.1

    model.rpn.pre_nms_top_n_train = 4000
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.pre_nms_top_n_test = 2000
    model.rpn.post_nms_top_n_test = 1000

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        cfg.num_classes,
    )

    model.roi_heads.detections_per_img = 200

    if getattr(cfg, "label_smoothing", 0.0) > 0:
        smooth_val = cfg.label_smoothing
        print(f"[Model] Label Smoothing enabled → {smooth_val}")
        model.roi_heads.fastrcnn_loss_func = torch.nn.CrossEntropyLoss(
            label_smoothing=smooth_val
        )
    else:
        model.roi_heads.fastrcnn_loss_func = torch.nn.CrossEntropyLoss()

    print("[RPN] Anchor sizes:", model.rpn.anchor_generator.sizes)
    print("[RPN] FG/BG IoU:", model.rpn.fg_iou_thresh, model.rpn.bg_iou_thresh)

    return model


# -------------------------
# Training / Evaluation
# -------------------------

def sanitize_targets_for_training(
    targets: List[Dict],
    device: torch.device,
) -> List[Dict]:
    """
    Ensure detection targets follow torchvision Faster R-CNN conventions.

    Fixes missing or malformed boxes/labels and moves all tensor values
    onto the correct device.

    Args:
        targets: List of raw target dictionaries.
        device: Device to move tensors to.

    Returns:
        List of sanitized target dictionaries.
    """
    clean = []
    for t in targets:
        boxes = t.get("boxes", None)
        labels = t.get("labels", None)
        if boxes is None or labels is None or boxes.ndim != 2 or boxes.size(-1) != 4:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels = torch.zeros((0,), dtype=torch.int64, device=device)
        item = dict(t)
        item["boxes"] = boxes
        item["labels"] = labels
        for k, v in item.items():
            if torch.is_tensor(v):
                item[k] = v.to(device, non_blocking=True)
        clean.append(item)
    return clean


def run_epoch_dynamic_sampler_if_needed(
    cfg: TrainConfig,
    dls: Dict[str, DataLoader],
    epoch: int,
) -> Dict[str, DataLoader]:
    """
    Refresh the dynamic WeightedRandomSampler at the start of an epoch.

    Recomputes balanced per-image weights for the training loader,
    logs raw/balanced distributions, and returns updated loaders.

    Args:
        cfg: Training configuration.
        dls: Dictionary of dataloaders with at least "train".
        epoch: 0-based epoch index.

    Returns:
        Updated dataloader dictionary (possibly unchanged).
    """
    ws_mode = str(getattr(cfg, "weight_sampling", "none")).lower().strip()
    if ws_mode != "dynamic":
        return dls

    json_path = Path(cfg.out_dir) / "dynamic_sampler_log.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = dls["train"].dataset

    if not hasattr(train_ds, "img_labels"):
        img_labels = []
        for img_id in train_ds.ids:
            ann_ids = train_ds.coco.getAnnIds(imgIds=[img_id])
            img_labels.append(1 if len(ann_ids) > 0 else 0)
        train_ds.img_labels = np.array(img_labels, dtype=np.int32)
    else:
        img_labels = np.asarray(train_ds.img_labels, dtype=np.int32)

    img_labels = np.array(img_labels, dtype=np.int32).flatten()
    n_total = len(img_labels)
    if n_total == 0:
        raise ValueError("[DynamicSampler] Dataset has 0 images — cannot build sampler.")

    n_smoke_raw = int(np.sum(img_labels == 1))
    n_back_raw = int(np.sum(img_labels == 0))
    ratio_smoke_raw = n_smoke_raw / n_total
    ratio_back_raw = n_back_raw / n_total

    w_smoke = 0.5 / max(1, n_smoke_raw)
    w_back = 0.5 / max(1, n_back_raw)
    weights = np.where(img_labels == 1, w_smoke, w_back).astype(np.float32)
    weights = np.ravel(weights)

    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    assert weights_tensor.ndim == 1 and len(weights_tensor) == n_total, (
        f"[DynamicSampler] Invalid weights shape: {weights_tensor.shape}, "
        f"expected ({n_total},)"
    )

    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=n_total,
        replacement=True,
    )

    dls["train"] = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=getattr(dls["train"], "collate_fn", None),
    )

    smoke_prob = np.sum(weights[img_labels == 1])
    back_prob = np.sum(weights[img_labels == 0])
    smoke_samples = int(round(smoke_prob * n_total))
    background_samples = int(round(back_prob * n_total))
    ratio_smoke_bal = smoke_samples / n_total
    ratio_back_bal = background_samples / n_total

    log_entry = {
        "run_name": getattr(cfg, "run_name", None),
        "project": str(Path(cfg.out_dir).parent),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_total": cfg.epochs,
        "batch_size": cfg.batch_size,
        "sampling_mode": "Dynamic WeightedRandomSampler",
        "records": [
            {
                "epoch": epoch + 1,
                "raw_distribution": {
                    "smoke_images": n_smoke_raw,
                    "background_images": n_back_raw,
                    "ratio_smoke": round(ratio_smoke_raw, 4),
                    "ratio_background": round(ratio_back_raw, 4),
                    "dataset_total_images": n_total,
                },
                "balanced_distribution": {
                    "smoke_samples": smoke_samples,
                    "background_samples": background_samples,
                    "ratio_smoke": round(ratio_smoke_bal, 4),
                    "ratio_background": round(ratio_back_bal, 4),
                    "total_samples": n_total,
                },
            }
        ],
    }

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            data[0]["records"].append(log_entry["records"][0])
        else:
            data = [data]
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(json_path, "w") as f:
            json.dump([log_entry], f, indent=2)

    print(f"[DynamicSampler] Epoch {epoch + 1}/{cfg.epochs}")
    print(
        f"    Raw distribution      → smoke={n_smoke_raw}, background={n_back_raw}, "
        f"ratio={ratio_smoke_raw:.3f}/{ratio_back_raw:.3f}, total={n_total}"
    )
    print(
        f"    Balanced distribution → smoke={smoke_samples}, "
        f"background={background_samples}, "
        f"ratio={ratio_smoke_bal:.3f}/{ratio_back_bal:.3f}, total={n_total}"
    )

    return dls


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    start_time: float,
    scaler: torch.cuda.amp.GradScaler,
    notebook_mode: bool = False,
    max_batches: Optional[int] = None,
    empty_cache: bool = True,
) -> Dict[str, float]:
    """
    Train Faster R-CNN for a single epoch.

    Performs forward, backward, and optimizer steps using mixed precision
    when enabled and returns averaged loss components.

    Args:
        model: Faster R-CNN model.
        optimizer: Optimizer for model parameters.
        data_loader: Training DataLoader.
        device: Device to run on.
        epoch: Current epoch index.
        total_epochs: Total number of epochs.
        start_time: Wall-clock training start time.
        scaler: Gradient scaler for AMP.
        notebook_mode: Whether to configure tqdm for notebooks.
        max_batches: Optional limit for number of batches in this epoch.
        empty_cache: Whether to clear CUDA cache after each batch.

    Returns:
        Dictionary with averaged loss components.
    """
    model.train()
    tqdm_fn = get_tqdm(notebook_mode)
    prog_bar = tqdm_fn(data_loader, dynamic_ncols=True, leave=True)

    running = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    n_batches = 0

    for i, (images, targets) in enumerate(prog_bar):
        if max_batches is not None and i >= max_batches:
            break

        images = [img.to(device, non_blocking=True) for img in images]
        targets = sanitize_targets_for_training(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for k in running.keys():
            running[k] += float(loss_dict[k].detach().cpu().item())
        n_batches += 1

        elapsed = time.time() - start_time
        global_step = epoch * len(data_loader) + (i + 1)
        avg_step = elapsed / max(1, global_step)
        remaining_steps = (total_epochs * len(data_loader)) - global_step
        eta_str = str(timedelta(seconds=int(avg_step * max(0, remaining_steps))))

        prog_bar.set_description(
            f"[Train {epoch + 1}/{total_epochs} | step {i + 1}/{len(data_loader)} | "
            f"loss={loss.item():.4f} | ETA {eta_str}]"
        )

        del images, targets, loss_dict, loss
        if empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()

    for k in running.keys():
        running[k] = running[k] / max(1, n_batches)
    return running


@torch.inference_mode()
def evaluate_coco_full(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    start_time: float,
    max_batches: Optional[int] = None,
    empty_cache: bool = True,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.45,
) -> Dict[str, float]:
    """
    Run full COCO evaluation over a validation or test split.

    Applies custom confidence and NMS thresholds, runs COCOeval, and
    returns mAP/AR metrics as well as average detections per image.

    Args:
        model: Faster R-CNN model.
        data_loader: DataLoader for the split.
        device: Device to run on.
        epoch: Current epoch index for display.
        total_epochs: Total number of epochs.
        start_time: Global training start time.
        max_batches: Optional limit on number of batches.
        empty_cache: Whether to clear CUDA cache after each batch.
        score_thresh: Confidence threshold for detections.
        nms_thresh: IoU threshold for NMS.

    Returns:
        Dictionary containing COCO metrics and average detections.
    """
    model.eval()

    core = model.module if hasattr(model, "module") else model
    if hasattr(core, "roi_heads"):
        core.roi_heads.score_thresh = score_thresh
        core.roi_heads.nms_thresh = nms_thresh
        print(f"[DEBUG] Set model thresholds → conf={score_thresh}, IoU(NMS)={nms_thresh}")

    results = []
    img_ids = []
    total_dets = 0

    tqdm_fn = get_tqdm()
    prog_bar = tqdm_fn(data_loader, dynamic_ncols=True, leave=False)

    with torch.inference_mode():
        for i, (images, targets) in enumerate(prog_bar):
            if max_batches is not None and i >= max_batches:
                break

            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                img_id = int(tgt["image_id"].item())
                img_ids.append(img_id)

                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()

                if score_thresh > 0:
                    keep = scores >= score_thresh
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                num_dets = boxes.shape[0]
                total_dets += num_dets
                if num_dets == 0:
                    continue

                xywh = boxes.copy()
                xywh[:, 2] -= xywh[:, 0]
                xywh[:, 3] -= xywh[:, 1]

                for b, s, lab in zip(xywh, scores, labels):
                    results.append(
                        {
                            "image_id": img_id,
                            "category_id": int(lab),
                            "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "score": float(s),
                        }
                    )

            del images, outputs
            if empty_cache and device.type == "cuda":
                torch.cuda.empty_cache()

    coco_gt = data_loader.dataset.coco
    if len(results) == 0:
        return {
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "AR_100": 0.0,
        }

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    avg_dets_per_img = total_dets / max(1, len(img_ids))
    stats = coco_eval.stats

    metrics = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
        "avg_dets_per_img": float(avg_dets_per_img),
    }

    elapsed = time.time() - start_time
    print(
        f"[Val {epoch + 1}/{total_epochs}] "
        f"mAP50={metrics['mAP_50']:.4f} "
        f"mAP50:95={metrics['mAP_50_95']:.4f} "
        f"AR100={metrics['AR_100']:.4f} "
        f"avg_dets={metrics['avg_dets_per_img']:.1f} "
        f"| Elapsed {elapsed / 60:.1f} min"
    )

    metrics["predictions"] = results
    return metrics


@torch.inference_mode()
def evaluate_coco_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    start_time: float,
    max_batches: Optional[int] = None,
    empty_cache: bool = True,
) -> Dict[str, float]:
    """
    Evaluate detection losses on a validation split.

    Runs the model in training mode to access loss_dict and returns
    averaged loss components, setting all COCO metrics to zero.

    Args:
        model: Faster R-CNN model.
        data_loader: Validation DataLoader.
        device: Device to run on.
        epoch: Current epoch index (for printing).
        total_epochs: Total number of epochs.
        start_time: Training start time (unused but kept for symmetry).
        max_batches: Optional maximum number of batches.
        empty_cache: Whether to clear CUDA cache after each batch.

    Returns:
        Dictionary with val_* loss keys and zeroed COCO metrics.
    """
    model.train()

    totals = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    n = 0

    tqdm_fn = get_tqdm()
    prog_bar = tqdm_fn(data_loader, dynamic_ncols=True, leave=False)

    for i, (images, targets) in enumerate(prog_bar):
        if max_batches is not None and i >= max_batches:
            break

        images = [img.to(device, non_blocking=True) for img in images]
        targets = sanitize_targets_for_training(targets, device)

        loss_dict = model(images, targets)
        for k in totals.keys():
            totals[k] += float(loss_dict[k].detach().cpu().item())
        n += 1

        del images, targets, loss_dict
        if empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()

    avg_losses = {f"val_{k}": v / max(1, n) for k, v in totals.items()}
    avg_total = sum(avg_losses.values())

    print(
        f"[ValLoss {epoch + 1}/{total_epochs}] "
        + " ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
        + f" total={avg_total:.4f}"
    )

    return {
        **avg_losses,
        "mAP_50_95": 0.0,
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "mAP_small": 0.0,
        "mAP_medium": 0.0,
        "mAP_large": 0.0,
        "AR_1": 0.0,
        "AR_10": 0.0,
        "AR_100": 0.0,
        "AR_small": 0.0,
        "AR_medium": 0.0,
        "AR_large": 0.0,
        "avg_dets_per_img": 0.0,
    }


# -------------------------
# Orchestration
# -------------------------

def train_rcnn(cfg: TrainConfig) -> Dict[str, float]:
    """
    Full end-to-end Faster R-CNN training and evaluation loop.

    Features:
      - DataLoader creation with static/dynamic weighting.
      - Model construction and ROI threshold tuning.
      - Warmup + cosine or warmup + plateau learning rate schedules.
      - Resumable checkpoints and early stopping.
      - COCO metrics on validation and test splits.
      - CSV logging, plots, and JSON prediction dumps.

    Args:
        cfg: Training configuration.

    Returns:
        Dictionary containing best validation mAP and final test metrics.
    """
    set_seed(cfg.seed, cfg.deterministic)
    ensure_outdir(cfg.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dls = make_dataloaders(cfg)
    EARLY_STOP_METRIC = "mAP_50_95"

    if getattr(cfg, "use_augmentations", False):
        visualize_augmentations_before_after(dls["train"].dataset, cfg.out_dir, n=6)

    config_file = cfg.out_dir / "config.json"
    env_file = cfg.out_dir / "env_info.json"
    stats_file = cfg.out_dir / "dataset_stats.json"

    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(cfg), f, indent=2, default=str)
        print(f"[INFO] Saved config.json → {config_file}")
    else:
        print("[INFO] Skipping config.json (already exists, resuming run)")

    if not env_file.exists():
        save_env(cfg, cfg.out_dir)
    else:
        print("[INFO] Skipping env_info.json (already exists, resuming run)")

    if not stats_file.exists():
        dataset_stats = save_dataset_stats(
            dls,
            cfg.out_dir,
            dataset_name=cfg.dataset_name,
            class_names=cfg.class_names,
        )
    else:
        print("[INFO] Skipping dataset_stats.json (already exists, resuming run)")
        with open(stats_file, "r") as f:
            dataset_stats = json.load(f)

    for split in ("train", "val", "test"):
        if split in dataset_stats:
            s = dataset_stats[split]
            images = s.get("images", 0)
            boxes = s.get("boxes", 0)
            per_class_images = s.get("per_class_images", {})
            bg_tot = per_class_images.get("background", 0)
            smoke_tot = per_class_images.get("smoke", 0)
            print(
                f"[DATASET] {split:<5} → images={images}, boxes={boxes}, "
                f"smoke={smoke_tot}, background={bg_tot} "
                f"({bg_tot / max(images, 1):.2%} background)"
            )

    model = get_fasterrcnn_model(cfg).to(device)

    core = model.module if hasattr(model, "module") else model
    core.roi_heads.score_thresh = cfg.eval_score_thresh
    core.roi_heads.nms_thresh = cfg.eval_nms_iou

    print(
        f"[INFO] ROI Heads thresholds set → "
        f"score_thresh={core.roi_heads.score_thresh}, "
        f"nms_thresh={core.roi_heads.nms_thresh}"
    )
    print(f"[INFO] ROI Heads loss function → {core.roi_heads.fastrcnn_loss_func}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    if cfg.lr_scheduler_type == "cosine":
        base_lr = cfg.learning_rate

        def lr_lambda(epoch: int) -> float:
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / float(max(1, cfg.warmup_epochs))
            total_cos_epochs = max(1, cfg.epochs - cfg.warmup_epochs)
            progress = (epoch - cfg.warmup_epochs + 1) / float(total_cos_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cfg.lrf + (1.0 - cfg.lrf) * cosine_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=-1,
        )
        print("[Scheduler] Using unified warm-up + cosine decay")
    elif cfg.lr_scheduler_type == "plateau":
        def warmup_lr_lambda(epoch: int) -> float:
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / float(max(1, cfg.warmup_epochs))
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lr_lambda,
        )

        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.lr_factor,
            patience=cfg.lr_plateau_patience,
            threshold=cfg.lr_plateau_threshold,
            min_lr=1e-6,
            verbose=True,
        )

        scheduler = WarmupPlateauScheduler(
            optimizer=optimizer,
            warmup_scheduler=warmup_scheduler,
            plateau_scheduler=plateau_scheduler,
            warmup_epochs=cfg.warmup_epochs,
        )
        print(
            "[Scheduler] Using Warm-up + ReduceLROnPlateau hybrid scheduler "
            f"(factor={cfg.lr_factor}, patience={cfg.lr_plateau_patience}, "
            f"threshold={cfg.lr_plateau_threshold})"
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.lr_scheduler_type}")

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    best_ckpt = cfg.out_dir / "fasterrcnn_best.pth"
    last_ckpt = cfg.out_dir / "fasterrcnn_last.pth"
    final_ckpt = cfg.out_dir / "fasterrcnn_final.pth"

    best_map = -float("inf")
    start_epoch = 0
    ckpt = None

    if cfg.resume_from == "last":
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location=device)
            print(f"Resuming from LAST checkpoint: {last_ckpt}")
        elif best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            print(f"LAST not found → Resuming from BEST checkpoint: {best_ckpt}")
        else:
            print("No checkpoints found → starting fresh training")
    elif cfg.resume_from == "best":
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            print(f"Resuming from BEST checkpoint: {best_ckpt}")
        else:
            print("BEST not found → starting fresh training")
    elif cfg.resume_from == "none":
        print("Resume disabled → starting fresh training")
    else:
        raise ValueError(f"Invalid resume_from option: {cfg.resume_from}")

    if ckpt is not None:
        print(f"[INFO] Restoring checkpoint from epoch {ckpt.get('epoch', '?')}")

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                print("[INFO] Scheduler state restored from checkpoint.")
            except Exception as e:
                print(f"[WARN] Could not restore scheduler state: {e}")
        else:
            print("[INFO] No scheduler state found in checkpoint — using fresh scheduler.")

        if cfg.amp and scaler and "scaler" in ckpt and ckpt["scaler"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("[INFO] AMP scaler restored.")
            except Exception as e:
                print(f"[WARN] Could not restore AMP scaler: {e}")

        last_epoch_ckpt = ckpt.get("epoch", -1)
        best_map = ckpt.get("best_map", -1.0)

        if last_epoch_ckpt >= 0:
            scheduler.last_epoch = last_epoch_ckpt
            print(f"[INFO] Resumed training at epoch {last_epoch_ckpt + 1}")

        start_epoch = last_epoch_ckpt + 1
        remaining = cfg.epochs - start_epoch
        no_improve = ckpt.get("no_improve", 0)

        last_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Resumed at epoch {start_epoch}, {remaining} epochs remaining "
            f"(Scheduler={cfg.lr_scheduler_type}, last LR={last_lr:.6f})"
        )
        print(f"[Resume] no_improve counter restored → {no_improve}")
        remaining_patience = max(0, cfg.patience - no_improve)
        print(f"[Resume] remaining early-stop patience: {remaining_patience}/{cfg.patience}")
    else:
        no_improve = 0

    history: List[Dict] = []
    log_csv = cfg.out_dir / "training_log.csv"
    if log_csv.exists():
        try:
            if pd is not None:
                df = pd.read_csv(log_csv)
                history = df.to_dict("records")
                print(f"[INFO] Resumed history from {log_csv} ({len(history)} rows)")
            else:
                with open(log_csv, "r") as f:
                    reader = csv.DictReader(f)
                    history = [row for row in reader]
                print(f"[INFO] Resumed history from {log_csv} ({len(history)} rows)")
        except Exception as e:
            print(f"[WARN] Could not load previous history: {e}")

    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    last_epoch = start_epoch - 1
    interrupted = False

    try:
        for epoch in range(start_epoch, cfg.epochs):
            lr = optimizer.param_groups[0]["lr"]

            if epoch < cfg.warmup_epochs:
                phase = "Warm-up"
            elif cfg.lr_scheduler_type.lower() == "cosine":
                phase = "Cosine"
            else:
                phase = "Plateau"

            print(
                f"[Scheduler] {phase} (LR={lr:.6f}, "
                f"epoch={epoch + 1}/{cfg.epochs})"
            )

            dls = run_epoch_dynamic_sampler_if_needed(cfg, dls, epoch)

            if hasattr(dls["train"], "sampler") and hasattr(dls["train"].sampler, "weights"):
                weights = dls["train"].sampler.weights
                labels = getattr(dls["train"].dataset, "labels", None)
                if labels is not None:
                    log_class_balance_rcnn(epoch, weights, labels, cfg.out_dir, cfg)

            epoch_start = time.time()

            train_losses = train_one_epoch(
                model,
                optimizer,
                dls["train"],
                device,
                epoch,
                cfg.epochs,
                start_time,
                scaler,
                max_batches=cfg.max_batches,
                empty_cache=cfg.empty_cache,
            )

            if device.type == "cuda":
                peak_train_mb = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
                print(f"Peak GPU mem (train): {peak_train_mb:.1f} MB")
            else:
                peak_train_mb = 0.0

            n_eval = max(1, int(cfg.eval_every_n))
            do_full_coco = (cfg.eval_mode == "coco") and (epoch % n_eval == 0)

            val_losses = evaluate_coco_loss(
                model,
                dls["val"],
                device,
                epoch,
                cfg.epochs,
                start_time,
                max_batches=cfg.max_batches,
                empty_cache=cfg.empty_cache,
            )

            val_metrics = dict(val_losses)

            if do_full_coco:
                print(f"[VAL] Epoch {epoch}: Running full COCO evaluation")
                coco_metrics = evaluate_coco_full(
                    model,
                    dls["val"],
                    device,
                    epoch,
                    cfg.epochs,
                    start_time,
                    max_batches=cfg.max_batches,
                    empty_cache=cfg.empty_cache,
                    score_thresh=float(cfg.eval_score_thresh),
                    nms_thresh=float(cfg.eval_nms_iou),
                )

                preds = coco_metrics.pop("predictions", None)
                if preds is not None:
                    preds_path = Path(cfg.out_dir) / f"predictions_val_epoch{epoch}.json"
                    try:
                        with open(preds_path, "w") as f:
                            json.dump(preds, f, indent=2)
                        avg_score = np.mean([p["score"] for p in preds]) if preds else 0
                        print(
                            f"[INFO] Saved {len(preds)} predictions to "
                            f"{preds_path.name} (avg_score={avg_score:.3f})"
                        )
                    except Exception as e:
                        print(f"[WARN] Could not save predictions.json: {e}")

                val_metrics.update(coco_metrics)

                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)) and v == -1.0:
                        val_metrics[k] = float("nan")

                val_metrics["val_mode"] = "coco"
            else:
                print(f"[VAL] Epoch {epoch}: Running loss-only validation")
                for k in [
                    "mAP_50", "mAP_50_95", "mAP_75",
                    "mAP_small", "mAP_medium", "mAP_large",
                    "AR_1", "AR_10", "AR_100",
                    "AR_small", "AR_medium", "AR_large",
                    "avg_dets_per_img",
                ]:
                    val_metrics[k] = float("nan")
                val_metrics["val_mode"] = "loss-only"

            if device.type == "cuda":
                peak_val_mb = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
                print(f"Peak GPU mem (val):   {peak_val_mb:.1f} MB")
            else:
                peak_val_mb = 0.0

            current_lr = optimizer.param_groups[0]["lr"]
            active_sched = cfg.lr_scheduler_type

            log_row = {
                "epoch": epoch,
                "lr": current_lr,
                "scheduler": active_sched,
                **{f"train_{k}": v for k, v in train_losses.items()},
                **val_metrics,
                "loss_type": cfg.loss_type,
                "train_peak_gpu_mem": peak_train_mb,
                "val_peak_gpu_mem": peak_val_mb,
                "time_sec": int(time.time() - epoch_start),
            }

            history.append(log_row)
            log_epoch_stats(history, cfg.out_dir)

            current_map = val_metrics.get(EARLY_STOP_METRIC, float("nan"))

            if best_map == -float("inf") or current_map > best_map:
                best_map = current_map
                no_improve = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_map": best_map,
                        "no_improve": no_improve,
                    },
                    best_ckpt,
                )

                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Saved new BEST checkpoint @ epoch {epoch} "
                    f"({EARLY_STOP_METRIC}={best_map:.4f})"
                )

                best_val_csv = cfg.out_dir / "best_val_metrics.csv"
                val_metrics["loss_type"] = cfg.loss_type
                val_metrics["epochs"] = cfg.epochs
                val_metrics["batch_size"] = cfg.batch_size
                val_metrics["epoch"] = epoch
                val_metrics["lr"] = current_lr

                preferred_order = [
                    "epoch", "lr",
                    "mAP_50", "mAP_50_95", "mAP_75",
                    "mAP_small", "mAP_medium", "mAP_large",
                    "AR_1", "AR_10", "AR_100",
                    "AR_small", "AR_medium", "AR_large",
                    "avg_dets_per_img",
                    "val_loss_classifier", "val_loss_box_reg",
                    "val_loss_objectness", "val_loss_rpn_box_reg",
                    "loss_type", "epochs", "batch_size",
                ]

                all_keys = set(val_metrics.keys())
                fieldnames = [k for k in preferred_order if k in all_keys] + [
                    k for k in sorted(all_keys) if k not in preferred_order
                ]

                row = {k: (val_metrics[k] if k in val_metrics else float("nan")) for k in fieldnames}

                if best_val_csv.exists():
                    with open(best_val_csv, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        old_rows = [r for r in reader]
                    old_epochs = {
                        int(r["epoch"]): r
                        for r in old_rows
                        if r.get("epoch", "") != ""
                    }
                    old_epochs[epoch] = row
                    merged = [old_epochs[e] for e in sorted(old_epochs.keys())]
                    with open(best_val_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(merged)
                else:
                    with open(best_val_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerow(row)

                print(
                    f"[INFO] best_val_metrics.csv updated at epoch {epoch} → "
                    f"{best_val_csv}"
                )
            else:
                no_improve += 1
                if epoch > 0:
                    print(
                        f"No {EARLY_STOP_METRIC} improvement for "
                        f"{no_improve} epoch(s)"
                    )

            if (epoch + 1) % 2 == 0:
                ckpt_path = os.path.join(
                    cfg.out_dir,
                    f"checkpoint_epoch_{epoch + 1:03d}.pth",
                )
                ckpt_tmp = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()
                    if (cfg.amp and scaler is not None)
                    else None,
                    "best_map": best_map,
                    "no_improve": no_improve,
                }
                torch.save(ckpt_tmp, ckpt_path)
                print(f"Saved checkpoint (every 2nd epoch) → {ckpt_path}")

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_map": best_map,
                    "no_improve": no_improve,
                },
                last_ckpt,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Saved LAST checkpoint @ epoch {epoch} → {last_ckpt}")

            if cfg.lr_scheduler_type == "plateau":
                val_score = val_metrics.get(EARLY_STOP_METRIC, None)
                scheduler.step(val_score)
            else:
                scheduler.step()

            last_epoch = epoch

            if no_improve >= cfg.patience:
                print(
                    f"Early stopping after {cfg.patience} epochs "
                    "without improvement."
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n[INFO] Training interrupted by user; skipping end-of-training "
            "artifacts (not at epoch boundary)."
        )

    if not interrupted:
        training_log_csv = cfg.out_dir / "training_log.csv"
        if training_log_csv.exists():
            plot_training_curves(training_log_csv, cfg.out_dir)

        print("\n[INFO] Final test evaluation (full COCO metrics)")
        test_metrics = evaluate_coco_full(
            model,
            dls["test"],
            device,
            epoch=min(cfg.epochs - 1, last_epoch),
            total_epochs=cfg.epochs,
            start_time=start_time,
            max_batches=None,
            empty_cache=cfg.empty_cache,
            score_thresh=float(cfg.eval_score_thresh),
            nms_thresh=cfg.eval_nms_iou,
        )

        preds = test_metrics.pop("predictions", None)
        if preds is not None:
            preds_path = Path(cfg.out_dir) / "predictions_test.json"
            try:
                with open(preds_path, "w") as f:
                    json.dump(preds, f, indent=2)
                avg_score = np.mean([p["score"] for p in preds]) if preds else 0
                print(
                    f"[INFO] Saved {len(preds)} test predictions to "
                    f"{preds_path.name} (avg_score={avg_score:.3f})"
                )
            except Exception as e:
                print(f"[WARN] Could not save predictions_test.json: {e}")

        print(
            f"[TEST] mAP@50={test_metrics['mAP_50']:.4f} | "
            f"mAP@50:95={test_metrics['mAP_50_95']:.4f} | "
            f"AR@100={test_metrics['AR_100']:.4f} | "
            f"avg_dets={test_metrics['avg_dets_per_img']:.1f}"
        )

        if device.type == "cuda":
            test_metrics["test_peak_gpu_mem"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

        test_lr = None
        if training_log_csv.exists():
            try:
                df = pd.read_csv(training_log_csv)
                row = df[df["epoch"] == last_epoch]
                if not row.empty and "lr" in row.columns:
                    test_lr = float(row.iloc[-1]["lr"])
            except Exception as e:
                print(f"[WARN] Could not read LR from training_log.csv: {e}")

        if test_lr is None:
            test_lr = float(optimizer.param_groups[0]["lr"])

        test_csv = cfg.out_dir / "test_results.csv"
        ensure_outdir(cfg.out_dir)
        test_metrics["lr_at_model_epoch"] = test_lr
        test_metrics["loss_type"] = cfg.loss_type
        test_metrics["model_epoch"] = int(last_epoch)
        test_metrics["batch_size"] = cfg.batch_size

        preferred_order = [
            "model_epoch", "lr_at_model_epoch",
            "mAP_50", "mAP_50_95", "mAP_75",
            "mAP_small", "mAP_medium", "mAP_large",
            "AR_1", "AR_10", "AR_100",
            "AR_small", "AR_medium", "AR_large",
            "avg_dets_per_img", "test_peak_gpu_mem",
            "loss_type", "epochs", "batch_size",
        ]

        all_keys = set(test_metrics.keys())
        fieldnames = [k for k in preferred_order if k in all_keys] + [
            k for k in sorted(all_keys) if k not in preferred_order
        ]

        row = {k: (test_metrics[k] if k in test_metrics else float("nan")) for k in fieldnames}

        if test_csv.exists():
            with open(test_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                old_rows = [r for r in reader]

            key = "model_epoch"
            old_map = {
                int(r[key]): r
                for r in old_rows
                if r.get(key, "") not in ("", None)
            }
            old_map[int(test_metrics["model_epoch"])] = row
            merged = [old_map[k] for k in sorted(old_map.keys())]

            with open(test_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(merged)
        else:
            with open(test_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerow(row)

        print(
            f"[INFO] test_results.csv updated for model_epoch {last_epoch} → "
            f"{test_csv}"
        )

        torch.save(
            {
                "epoch": last_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_map": best_map,
            },
            final_ckpt,
        )
        print(f"Saved FINAL model → {final_ckpt}")

    return {
        "best_map50_95": best_map,
        **(test_metrics if not interrupted else {}),
    }


# -------------------------
# Script entry point
# -------------------------

if __name__ == "__main__":
    PROJECT_DIR = Path.cwd().parent
    sys.path.append(str(PROJECT_DIR))
    SRC_DIR = PROJECT_DIR / "src"
    sys.path.append(str(SRC_DIR))

    CONFIG_DIR = PROJECT_DIR / "configs" / "jobs_rcnn_resplit_new"

    def parse_value(v: Any) -> Any:
        """
        Parse simple scalar types from YAML string values.

        Converts typical boolean, None, int, and float representations.

        Args:
            v: Value from YAML.

        Returns:
            Parsed Python type.
        """
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in {"true", "yes"}:
                return True
            if v_lower in {"false", "no"}:
                return False
            if v_lower in {"none", "null", ""}:
                return None
            try:
                if "." in v_lower:
                    return float(v)
                return int(v)
            except ValueError:
                return v
        return v

    yaml_files = sorted(CONFIG_DIR.glob("*.yaml"))

    if yaml_files:
        print(f"Found {len(yaml_files)} YAML configs in {CONFIG_DIR}")
        for config_file in yaml_files:
            print(f"\nStarting batch job from {config_file.name}")
            with open(config_file, "r") as f:
                cfg_dict = yaml.safe_load(f)
            cfg_dict = {k: parse_value(v) for k, v in cfg_dict.items()}

            cfg = TrainConfig(
                images_root_train=(PROJECT_DIR / cfg_dict.get("images_root_train")).resolve(),
                images_root_val=(PROJECT_DIR / cfg_dict.get("images_root_val")).resolve(),
                images_root_test=(PROJECT_DIR / cfg_dict.get("images_root_test")).resolve(),
                train_json=(PROJECT_DIR / cfg_dict.get("train_json")).resolve(),
                val_json=(PROJECT_DIR / cfg_dict.get("val_json")).resolve(),
                test_json=(PROJECT_DIR / cfg_dict.get("test_json")).resolve(),
                out_dir=PROJECT_DIR / "results" / "rcnn_batch" / config_file.stem,
                epochs=int(cfg_dict.get("epochs", 50)),
                batch_size=int(cfg_dict.get("batch", 6)),
                num_workers=int(cfg_dict.get("num_workers", 16)),
                momentum=float(cfg_dict.get("momentum", 0.9)),
                weight_decay=float(cfg_dict.get("weight_decay", 1e-4)),
                learning_rate=float(cfg_dict.get("lr0", 0.0025)),
                lr_milestones=(3, 6),
                max_batches=None,
                lr_scheduler_type=cfg_dict.get("scheduler", "cosine"),
                lr_factor=float(cfg_dict.get("lr_factor", 0.7)),
                use_augmentations=cfg_dict.get("use_aug", False),
                seed=int(cfg_dict.get("seed", 42)),
                num_classes=2,
                patience=int(cfg_dict.get("patience", 8)),
                eval_mode="coco",
                eval_every_n=1,
                loss_type=cfg_dict.get("loss_type", "ce"),
                empty_cache=True,
                resume_from="last",
                label_smoothing=float(cfg_dict.get("label_smoothing", 0.1)),
                weight_sampling=str(cfg_dict.get("weight_sampling", "false")),
                warmup_epochs=int(cfg_dict.get("warmup_epochs", 3)),
                eval_nms_iou=float(cfg_dict.get("eval_iou", 0.45)),
                eval_score_thresh=float(cfg_dict.get("eval_conf", 0.05)),
            )

            out_dir = Path(cfg.out_dir)
            done_flag = out_dir / "done.flag"
            metrics_file = out_dir / "metrics_test.json"

            if out_dir.exists():
                if done_flag.exists() or metrics_file.exists():
                    print(f"[SKIP] {config_file.name}: already completed — skipping.")
                    continue

            print(f"[ALIGN] LR base={cfg.learning_rate:.6f}, lrf={cfg.lrf}")

            try:
                results = train_rcnn(cfg)
                print("Finished:", config_file.stem)
                print("Test results:", results)
                (out_dir / "done.flag").write_text("OK\n")
            except KeyboardInterrupt:
                print(f"[INTERRUPTED] Training manually stopped for {config_file.stem}")
            except Exception as e:
                print(f"[ERROR] Training failed for {config_file.stem}: {e}")
    else:
        print("No YAML configs found — running single default training...")

        cfg = TrainConfig(
            images_root_train=PROJECT_DIR.parent / "data" / "pyro-sdis" / "images",
            images_root_val=PROJECT_DIR.parent / "data" / "pyro-sdis" / "images",
            images_root_test=PROJECT_DIR.parent / "data" / "pyro-sdis" / "images",
            train_json=PROJECT_DIR / "data" / "pyro_sdis" / "splits" / "pyro_sdis_train.json",
            val_json=PROJECT_DIR / "data" / "pyro_sdis" / "splits" / "pyro_sdis_val.json",
            test_json=PROJECT_DIR / "data" / "pyro_sdis" / "splits" / "pyro_sdis_test.json",
            out_dir=PROJECT_DIR / "results" / "rcnn_baseline" / "smoke_plateau",
            epochs=15,
            max_batches=20,
            batch_size=4,
            num_workers=16,
            momentum=0.9,
            weight_decay=1e-4,
            learning_rate=0.0025,
            lr_milestones=(3, 6),
            lr_scheduler_type="plateau",
            lr_factor=0.7,
            use_augmentations=True,
            amp=True,
            seed=42,
            num_classes=2,
            patience=20,
            eval_mode="coco",
            eval_every_n=1,
            loss_type="ce",
            empty_cache=True,
            resume_from="last",
            label_smoothing=0.1,
            weight_sampling="false",
            warmup_epochs=3,
        )

        print(f"[ALIGN] LR base={cfg.learning_rate:.6f}, lrf={cfg.lrf}")
        results = train_rcnn(cfg)
        print("Test results:", results)
