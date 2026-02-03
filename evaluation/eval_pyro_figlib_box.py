from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import re
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset
from ultralytics import YOLO
from dataclasses import dataclass
from training.train_faster_rcnn import get_fasterrcnn_model

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

MODEL_TYPE = "yolo"   # "yolo" or "rcnn"
DATASET_TYPE = "pyro"  # "pyro" or "figlib"

DO_YOLO_SANITY_PLOT = False

BASE_RESULTS = Path(r"D:\github\XAI-wildfire-smoke-detection\final_results")

VAL_IMG_DIR = r"D:\github\data\figlib_with_bounding_boxes\images"
VAL_JSON    = Path(r"D:\github\data\figlib_with_bounding_boxes\figlib_coco_annotations.json")

OUT_DIR = Path("../final_results/evaluation_thesis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_CACHE_DIR = BASE_RESULTS / "yolo_cache"
YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RCNN_CACHE_DIR = BASE_RESULTS / "rcnn_cache"
RCNN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV  = OUT_DIR / f"{MODEL_TYPE}_coco_longform.csv"
OUT_PLOT = OUT_DIR / f"{MODEL_TYPE}_stage_comparison.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})


@dataclass
class EvalConfig:
    pretrained: bool = False
    num_classes: int = 2
    label_smoothing: float = 0.1


class SmokeValDataset(CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)
        self.tf = T.ToTensor()

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        img = self.tf(img)
        target = {
            "image_id": torch.tensor(self.ids[idx]),
            "annotations": ann
        }
        return img, target


class FIgLibDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset for FIgLib inference-only evaluation.
    No annotations are assumed.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.img_paths = sorted([
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in [".jpg", ".png"]
        ])
        self.tf = T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.tf(img), str(img_path)





def coco_stats_to_dict(stats):
    return {
        "AP50_95": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR_1": stats[6],
        "AR_10": stats[7],
        "AR_100": stats[8],
        "AR_small": stats[9],
        "AR_medium": stats[10],
        "AR_large": stats[11],
    }


def yolo_cache_path(stage: str, model_name: str) -> Path:
    return YOLO_CACHE_DIR / stage / model_name / "predictions.pkl"



def rcnn_cache_path(stage: str, model_name: str) -> Path:
    return RCNN_CACHE_DIR / stage / model_name / "predictions.pkl"


def save_rcnn_cache(stage, model_name, preds):
    path = rcnn_cache_path(stage, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preds, f)


def load_rcnn_cache(stage, model_name):
    path = rcnn_cache_path(stage, model_name)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def coco_collate_fn(batch):
    return tuple(zip(*batch))


def figlib_collate_fn(batch):
    # batch is a list of (img_tensor, path_str)
    images, paths = zip(*batch)
    return list(images), list(paths)



def load_frcnn(ckpt_path, device):

    cfg = EvalConfig()
    model = get_fasterrcnn_model(cfg)

    state = torch.load(ckpt_path, map_location="cpu")

    # Handle common checkpoint formats
    if "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model

# ======================================================
# MAIN
# ======================================================
def run_evaluation():

    # ---------------- DATASET ----------------
    if DATASET_TYPE == "figlib":
        dataset = FIgLibDataset(VAL_IMG_DIR)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=figlib_collate_fn)
    else:
        dataset = SmokeValDataset(VAL_IMG_DIR, VAL_JSON)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=coco_collate_fn
        )

    rows = []

    # ==================================================
    # YOLO
    # ==================================================
    if MODEL_TYPE == "yolo":

        for stage_dir in sorted((BASE_RESULTS / "yolo").glob("stage*")):
            stage = stage_dir.name

            for model_dir in stage_dir.iterdir():
                ckpt = model_dir / "weights" / "best.pt"
                if not ckpt.exists():
                    continue

                print(f"📦 YOLO | {stage} | {model_dir.name}")
                yolo = YOLO(str(ckpt))
                preds = []

                if DATASET_TYPE == "figlib":
                    img_paths = [str(p) for p in dataset.img_paths]
                    img_ids = list(range(len(img_paths)))
                else:
                    img_ids = dataset.ids
                    img_paths = [
                        str(Path(VAL_IMG_DIR) / dataset.coco.imgs[i]["file_name"])
                        for i in img_ids
                    ]

                BATCH = 8
                for i in range(0, len(img_paths), BATCH):
                    batch_paths = img_paths[i:i+BATCH]
                    batch_ids = img_ids[i:i+BATCH]

                    outs = yolo.predict(batch_paths, imgsz=640, conf=0.001, iou=0.7, verbose=False)

                    for img_id, path, out in zip(batch_ids, batch_paths, outs):
                        if out.boxes is None:
                            preds.append({
                                "image_id": img_id,
                                "image_path": path,
                                "event_id": Path(path).parent.name,
                                "boxes": np.zeros((0,4), np.float32),
                                "scores": np.zeros((0,), np.float32),
                                "labels": np.zeros((0,), np.int64)
                            })
                        else:
                            preds.append({
                                "image_id": img_id,
                                "image_path": path,
                                "event_id": Path(path).parent.name,
                                "boxes": out.boxes.xyxy.cpu().numpy(),
                                "scores": out.boxes.conf.cpu().numpy(),
                                "labels": out.boxes.cls.long().cpu().numpy() + 1
                            })

                # save cache
                out_dir = YOLO_CACHE_DIR / stage / model_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "predictions.pkl", "wb") as f:
                    pickle.dump(preds, f)

                # evaluate only if pyro
                metrics = {}
                if DATASET_TYPE == "pyro":
                    metrics = evaluate_coco_from_cache(preds, dataset)

                rows.append({
                    "stage": stage,
                    "model": model_dir.name,
                    **metrics
                })

                del yolo
                torch.cuda.empty_cache()

    # ==================================================
    # Faster R-CNN
    # ==================================================
    elif MODEL_TYPE == "rcnn":

        for stage_dir in sorted((BASE_RESULTS / "rcnn").glob("stage*")):
            stage = stage_dir.name

            for model_dir in stage_dir.iterdir():
                ckpt = model_dir / "fasterrcnn_best.pth"
                if not ckpt.exists():
                    continue

                cache = load_rcnn_cache(stage, model_dir.name)

                if cache is not None:
                    print(f"📂 RCNN | {stage} | {model_dir.name} → using cached predictions")
                    preds = cache

                else:
                    print(f"📦 RCNN | {stage} | {model_dir.name} → running inference")
                    model = load_frcnn(ckpt, DEVICE)
                    preds = []

                    for batch in loader:

                        if DATASET_TYPE == "figlib":
                            images, paths = batch
                            images = [img.to(DEVICE) for img in images]

                            with torch.no_grad():
                                outputs = model(images)

                            for path, out in zip(paths, outputs):
                                preds.append({
                                    "image_id": path,
                                    "image_path": path,
                                    "event_id": Path(path).parent.name,
                                    "boxes": out["boxes"].detach().cpu().numpy(),
                                    "scores": out["scores"].detach().cpu().numpy(),
                                    "labels": out["labels"].detach().cpu().numpy(),
                                })

                        else:  # Pyro-SDIS
                            images, targets = batch
                            images = [img.to(DEVICE) for img in images]

                            with torch.no_grad():
                                outputs = model(images)

                            for tgt, out in zip(targets, outputs):
                                preds.append({
                                    "image_id": tgt["image_id"].item(),
                                    "image_path": None,
                                    "event_id": None,
                                    "boxes": out["boxes"].detach().cpu().numpy(),
                                    "scores": out["scores"].detach().cpu().numpy(),
                                    "labels": out["labels"].detach().cpu().numpy(),
                                })

                    save_rcnn_cache(stage, model_dir.name, preds)

                    del model
                    torch.cuda.empty_cache()

                # ✅ ALWAYS evaluate + append
                metrics = {}
                if DATASET_TYPE == "pyro":
                    metrics = evaluate_coco_from_cache(preds, dataset)

                rows.append({
                    "stage": stage,
                    "model": model_dir.name,
                    **metrics
                })


    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / f"{MODEL_TYPE}_{DATASET_TYPE}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"💾 Saved → {out_csv}")

    return df


def evaluate_coco_from_cache(preds, dataset):

    coco = get_coco_api_from_dataset(dataset)
    evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    results = {}
    for p in preds:
        results[p["image_id"]] = {
            "boxes": torch.tensor(p["boxes"], dtype=torch.float32),
            "scores": torch.tensor(p["scores"], dtype=torch.float32),
            "labels": torch.tensor(p["labels"], dtype=torch.long),
        }

    evaluator.update(results)

    # REQUIRED sequence
    evaluator.accumulate()
    evaluator.summarize()   # ← THIS WAS MISSING

    stats = evaluator.coco_eval["bbox"].stats
    return coco_stats_to_dict(stats)


def plot_stage_comparison(df):

    METRIC_COLS = [
        "AP50_95",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
    ]

    # --------- safety checks ---------
    required_cols = {"stage", "model"} | set(METRIC_COLS)
    if df.empty or not required_cols.issubset(df.columns):
        raise ValueError(
            f"Cannot plot stage comparison. "
            f"DF shape={df.shape}, columns={df.columns.tolist()}"
        )

    # --------- aggregate ---------
    grouped = (
        df
        .groupby(["stage", "model"])[METRIC_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=False)

    metrics = [
        ("AP50_95", "mAP@50–95"),
        ("AP_small", "AP (small)"),
        ("AP_medium", "AP (medium)"),
        ("AP_large", "AP (large)")
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for stage in grouped["stage"].unique():
            sub = grouped[grouped["stage"] == stage]

            ax.bar(
                sub["model"],
                sub[(metric, "mean")],
                yerr=sub[(metric, "std")],
                capsize=3,
                alpha=0.85,
                label=stage
            )

        ax.set_title(title)
        ax.set_ylabel("Average Precision")
        ax.set_xticklabels(sub["model"], rotation=45, ha="right")

    axes[0].legend(title="Training stage")
    plt.tight_layout()
    plt.savefig(OUT_PLOT)
    plt.close()

    print(f"📊 Saved plot → {OUT_PLOT}")


def plot_yolo_vs_gt(image_id, dataset, preds, img_dir, save_path=None):
    """
    Visualize GT vs YOLO predictions for a single image.
    """

    img_info = dataset.coco.imgs[image_id]
    img_path = Path(img_dir) / img_info["file_name"]
    img = Image.open(img_path).convert("RGB")

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")

    # ---- GT boxes (green)
    ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
    anns = dataset.coco.loadAnns(ann_ids)

    for ann in anns:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

    # ---- YOLO boxes (red)
    pred = next(p for p in preds if p["image_id"] == image_id)

    for box, score in zip(pred["boxes"], pred["scores"]):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1, f"{score:.2f}", color="red", fontsize=8)

    ax.set_title("GT (green) vs YOLO (red)")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":

    df = run_evaluation()

    if DATASET_TYPE == "pyro":
        plot_stage_comparison(df)
