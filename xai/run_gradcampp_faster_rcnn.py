"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions in the CONFIG section below
(e.g., IMG_DIR, COCO_JSON, BASE_OUT_DIR) to match your local directory structure.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import json
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn.functional as F
from pycocotools.coco import COCO

from utils.rcnn_model import XaiTrainConfig, get_fasterrcnn_model_xai

# =======
# CONFIG
# =======

IMG_DIR = Path(
    "/path/to/figlib/images"
)

COCO_JSON = Path(
    "/path/to/annotations.json"
)

CKPT_PATH = Path(
    "/path/to/trained_models"
)

PREDICTIONS_PKL = Path(
    "/path/to/prediction_cache"
)

OUT_DIR = Path(
    "/path/to/output"
)

MASTER_SAMPLE_JSON = Path(
    "/path/to/xai_cross_domain_master_sample.json"
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42

NUM_CLASSES = 2
SMOKE_CLASS = 1

CONF_THRESH = 0.30

CAM_TOP_PCT = 0.15
OVERLAY_ALPHA = 0.45

# =========================================================
# Utils
# =========================================================
def load_predictions(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        preds = pickle.load(f)
    return preds

def index_preds_by_image_id(preds):
    by_id = {}
    for p in preds:
        if "image_id" in p:
            by_id[int(p["image_id"])] = p
    return by_id

def has_pred_smoke_cached(p, conf_thresh: float) -> bool:
    scores = np.asarray(p.get("scores", []), dtype=np.float32)
    labels = np.asarray(p.get("labels", []), dtype=np.int64)
    keep = (labels == SMOKE_CLASS) & (scores >= conf_thresh)
    return bool(keep.any())

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_model_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return x.to(device)

def load_coco_gt(coco_json: Path) -> Tuple[Dict[int, str], Dict[int, np.ndarray]]:
    coco = COCO(str(coco_json))
    id2file = {img_id: coco.imgs[img_id]["file_name"] for img_id in coco.imgs.keys()}

    gt_boxes = {}
    for img_id in coco.imgs.keys():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])

        gt_boxes[img_id] = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

    return id2file, gt_boxes


def cam_threshold_top_pct(cam: np.ndarray, top_pct: float) -> np.ndarray:
    flat = cam.flatten()
    if flat.size == 0:
        return np.zeros_like(cam, dtype=bool)
    thr = np.quantile(flat, 1.0 - top_pct)
    return cam >= thr

def iou_mask_boxes(mask: np.ndarray, boxes_xyxy: np.ndarray) -> float:
    if boxes_xyxy.shape[0] == 0:
        return float("nan")
    H, W = mask.shape[:2]
    if mask.sum() <= 0:
        return 0.0

    gt_mask = np.zeros((H, W), dtype=bool)
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1i = max(0, int(np.floor(x1)))
        y1i = max(0, int(np.floor(y1)))
        x2i = min(W, int(np.ceil(x2)))
        y2i = min(H, int(np.ceil(y2)))
        if x2i > x1i and y2i > y1i:
            gt_mask[y1i:y2i, x1i:x2i] = True

    inter = float((mask & gt_mask).sum())
    union = float((mask | gt_mask).sum()) + 1e-9
    return inter / union

def pointing_game(cam: np.ndarray, boxes_xyxy: np.ndarray) -> float:
    if boxes_xyxy.shape[0] == 0:
        return float("nan")
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    for (x1, y1, x2, y2) in boxes_xyxy:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 1.0
    return 0.0

def energy_inside(cam: np.ndarray, boxes_xyxy: np.ndarray) -> float:
    if boxes_xyxy.shape[0] == 0:
        return float("nan")
    H, W = cam.shape[:2]
    tot = float(cam.sum()) + 1e-9
    inside = 0.0
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1i = max(0, int(np.floor(x1)))
        y1i = max(0, int(np.floor(y1)))
        x2i = min(W, int(np.ceil(x2)))
        y2i = min(H, int(np.ceil(y2)))
        if x2i > x1i and y2i > y1i:
            inside += float(cam[y1i:y2i, x1i:x2i].sum())
    return inside / tot

def largest_gt_box(gtb: np.ndarray) -> Optional[np.ndarray]:
    if gtb.shape[0] == 0:
        return None
    areas = (gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])
    return gtb[int(np.argmax(areas))].astype(np.float32)

def compute_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    xA = max(float(a[0]), float(b[0]))
    yA = max(float(a[1]), float(b[1]))
    xB = min(float(a[2]), float(b[2]))
    yB = min(float(a[3]), float(b[3]))

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    areaB = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / (areaA + areaB - inter + 1e-9))


# =========================================================
# Loader (REUSES YOUR get_fasterrcnn_model)
# =========================================================
def load_frcnn_from_ckpt(ckpt_path: Path, device: str) -> torch.nn.Module:
    cfg = XaiTrainConfig(
        pretrained=True,
        num_classes=NUM_CLASSES,
        label_smoothing=0.1,
    )
    model = get_fasterrcnn_model_xai(cfg)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# =========================================================
# Grad-CAM++ (instance-conditioned on one RoI logit)
# =========================================================
class GradCAMPP:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out
        self.gradients = None
        if hasattr(out, "requires_grad") and out.requires_grad:
            out.register_hook(lambda g: setattr(self, "gradients", g))

    @torch.enable_grad()
    def forward_roi_logits(self, img_tensor: torch.Tensor):
        m = self.model
        images_list = [img_tensor]
        images, _ = m.transform(images_list, targets=None)

        features = m.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, _ = m.rpn(images, features, targets=None)
        box_features = m.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = m.roi_heads.box_head(box_features)
        class_logits, box_regression = m.roi_heads.box_predictor(box_features)

        decoded = m.roi_heads.box_coder.decode(box_regression, proposals)
        decoded_boxes = decoded

        return class_logits, decoded_boxes, images

    def generate_for_roi(self, img_tensor: torch.Tensor, orig_rgb: np.ndarray, roi_index: int, target_class: int):
        self.activations = None
        self.gradients = None

        class_logits, decoded_boxes, images = self.forward_roi_logits(img_tensor)

        if roi_index < 0 or roi_index >= class_logits.shape[0]:
            raise RuntimeError(f"roi_index out of range: {roi_index} / {class_logits.shape[0]}")

        score = class_logits[roi_index, target_class]

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        A = self.activations
        dA = self.gradients if self.gradients is not None else getattr(A, "grad", None)
        if dA is None:
            raise RuntimeError("No gradients captured at target layer.")

        dA2 = dA * dA
        dA3 = dA2 * dA

        eps = 1e-9
        sum_A_dA3 = (A * dA3).sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * dA2 + sum_A_dA3
        denom = torch.where(denom != 0.0, denom, torch.full_like(denom, eps))

        alpha = dA2 / denom
        relu_dA = F.relu(dA)
        weights = (alpha * relu_dA).sum(dim=(2, 3), keepdim=True)

        cam = (weights * A).sum(dim=1)
        cam = F.relu(cam)[0].detach().float().cpu().numpy()

        cam = cv2.GaussianBlur(cam, (5, 5), 0)
        low, high = np.percentile(cam, [5, 99])
        cam = np.clip((cam - low) / (high - low + 1e-6), 0, 1)

        cam -= cam.min()
        cam /= (cam.max() + 1e-6)

        resized_h, resized_w = images.tensors.shape[-2], images.tensors.shape[-1]
        cam_resized = cv2.resize(cam, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        oh, ow = orig_rgb.shape[:2]
        cam_orig = cv2.resize(cam_resized, (ow, oh), interpolation=cv2.INTER_LINEAR)

        cam_orig = cam_orig ** 2.0
        cam_orig = cam_orig / (cam_orig.max() + 1e-6)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_orig), cv2.COLORMAP_JET)
        overlay = (OVERLAY_ALPHA * heatmap[..., ::-1] + (1.0 - OVERLAY_ALPHA) * orig_rgb).astype(np.uint8)

        return cam_orig, overlay, float(score.detach().cpu())


# =========================================================
# RoI selection
# =========================================================
@torch.no_grad()
def choose_roi_index_tp_fp(class_logits: torch.Tensor) -> int:
    probs = F.softmax(class_logits, dim=1)[:, SMOKE_CLASS]
    return int(torch.argmax(probs).item())

@torch.no_grad()
def choose_roi_index_fn_resized(decoded_boxes: torch.Tensor, gt_box_resized: np.ndarray) -> int:
    boxes = decoded_boxes.detach().cpu().numpy().astype(np.float32)
    ious = [compute_iou_xyxy(b, gt_box_resized) for b in boxes]
    return int(np.argmax(ious))

def resize_box_xyxy(box: np.ndarray, orig_hw: Tuple[int, int], resized_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = orig_hw
    rh, rw = resized_hw
    sx = rw / max(ow, 1)
    sy = rh / max(oh, 1)
    x1, y1, x2, y2 = box
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


# =========================================================
# NEW: load master sample file
# =========================================================
def load_master_samples(json_path: Path) -> Dict[int, Dict]:
    """
    Expected format:
    [
      {
        "image_id": 284,
        "file_name": "...jpg",
        "n_gt_boxes": 1,
        "has_smoke": 1,
        "pred_smoke_ref": 1
      },
      ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Master sample JSON must be a list of dicts.")

    by_id = {}
    for d in data:
        img_id = int(d["image_id"])
        by_id[img_id] = {
            "file_name": d.get("file_name"),
            "n_gt_boxes": int(d.get("n_gt_boxes", 0)),
            "has_smoke": int(d.get("has_smoke", 0)),
            "pred_smoke_ref": int(d.get("pred_smoke_ref", 0)),
        }
    return by_id


# =========================================================
# Main
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "overlays")
    ensure_dir(OUT_DIR / "cams")

    print(f"[INFO] Device: {DEVICE}")

    # Ground truth
    id2file, gt_boxes = load_coco_gt(COCO_JSON)
    all_image_ids = set(id2file.keys())
    print(f"[INFO] COCO images: {len(all_image_ids)}")

    # Model
    model = load_frcnn_from_ckpt(CKPT_PATH, device=DEVICE)
    print("[INFO] Loaded Faster R-CNN from ckpt via get_fasterrcnn_model_xai().")

    target_layer = model.backbone.body.layer4
    cam_engine = GradCAMPP(model, target_layer=target_layer)

    # Cached predictions
    preds = load_predictions(PREDICTIONS_PKL)
    preds_by_id = index_preds_by_image_id(preds)

    # Load master sample list
    master_samples = load_master_samples(MASTER_SAMPLE_JSON)
    sampled_ids = [i for i in master_samples.keys() if i in all_image_ids]
    print(f"[INFO] Using {len(sampled_ids)} images from master sample file")

    # Build gt_smoke, pred_smoke, group_of from master file
    gt_smoke = {}
    pred_smoke = {}
    group_of = {}

    for img_id in sampled_ids:
        meta = master_samples[img_id]

        gt_smoke[img_id] = int(meta["has_smoke"])
        pred_smoke[img_id] = int(meta["pred_smoke_ref"])

        if gt_smoke[img_id] == 1 and pred_smoke[img_id] == 1:
            g = "TP"
        elif gt_smoke[img_id] == 0 and pred_smoke[img_id] == 1:
            g = "FP"
        elif gt_smoke[img_id] == 1 and pred_smoke[img_id] == 0:
            g = "FN"
        else:
            g = "TN"
        group_of[img_id] = g

    rows = []
    processed = 0

    for img_id in sampled_ids:
        fn = id2file[img_id]
        rgb = read_rgb(IMG_DIR / fn)
        x = to_model_tensor(rgb, DEVICE)

        group = group_of[img_id]
        gtb = gt_boxes[img_id]
        p = preds_by_id.get(img_id, None)

        # Forward once to get RoI logits and decoded boxes (Faster R-CNN space)
        class_logits, decoded_boxes, images = cam_engine.forward_roi_logits(x)
        resized_h, resized_w = images.tensors.shape[-2], images.tensors.shape[-1]
        oh, ow = rgb.shape[:2]

        target_src = "none"
        target_box = None
        target_best_gt_idx = -1
        target_best_gt_iou = np.nan

        # ---------------- target detection selection (align with YOLO) ----------------
        # Use cached predictions (in original image coords) to pick a smoke box
        if p is not None:
            boxes_pred = np.asarray(p.get("boxes", np.zeros((0, 4), np.float32)), dtype=np.float32)
            scores_pred = np.asarray(p.get("scores", np.zeros((0,), np.float32)), dtype=np.float32)
            labels_pred = np.asarray(p.get("labels", np.zeros((0,), np.int64)), dtype=np.int64)

            # smoke + score threshold, same logic as YOLO
            mask_smoke = (labels_pred == SMOKE_CLASS) & (scores_pred >= CONF_THRESH)

            if group in ("TP", "FP") and mask_smoke.any():
                # best smoke prediction by score
                idx_sm = np.where(mask_smoke)[0]
                best_local = int(idx_sm[np.argmax(scores_pred[idx_sm])])
                target_box = boxes_pred[best_local]
                target_src = "pred_box"
                det_score = float(scores_pred[best_local])
            else:
                det_score = np.nan
        else:
            boxes_pred = np.zeros((0, 4), np.float32)
            det_score = np.nan

        # FN or fallback: use largest GT box if available
        if target_box is None and gtb.shape[0] > 0:
            areas = (gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])
            j = int(np.argmax(areas))
            target_box = gtb[j]
            target_src = "gt_box"

        # Still nothing: fallback to image center (no explicit box)
        if target_box is None:
            target_src = "image_center"

        # ---------------- RoI index selection (Faster R-CNN space) ----------------
        # For TP/FP: match decoded_boxes to the chosen target_box (resized)
        boxes_all = decoded_boxes.detach().cpu().numpy().astype(np.float32)

        if target_src in ("pred_box", "gt_box") and boxes_all.shape[0] > 0:
            box_resized = resize_box_xyxy(target_box, (oh, ow), (resized_h, resized_w))
            ious = [compute_iou_xyxy(b, box_resized) for b in boxes_all]
            roi_index = int(np.argmax(ious))
        else:
            # Fallback: highest smoke probability RoI
            roi_index = choose_roi_index_tp_fp(class_logits)

        # ---------------- Grad-CAM++ generation ----------------
        cam, overlay, score_logit = cam_engine.generate_for_roi(
            x, rgb, roi_index=roi_index, target_class=SMOKE_CLASS
        )

        # ---------------- per-target GT matching (IoU) ----------------
        if gtb.shape[0] > 0 and target_box is not None:
            ious_to_gt = np.array([compute_iou_xyxy(target_box, g) for g in gtb], dtype=np.float32)
            target_best_gt_idx = int(np.argmax(ious_to_gt))
            target_best_gt_iou = float(ious_to_gt[target_best_gt_idx])
        else:
            target_best_gt_idx = -1
            target_best_gt_iou = np.nan

        # ---------------- XAI metrics (same names as YOLO) ----------------
        cam_iou = np.nan
        cam_pg = np.nan
        cam_ein = np.nan
        if gtb.shape[0] > 0:
            mask = cam_threshold_top_pct(cam, CAM_TOP_PCT)
            cam_iou = iou_mask_boxes(mask, gtb)
            cam_pg = pointing_game(cam, gtb)
            cam_ein = energy_inside(cam, gtb)

        # ---------------- saving outputs ----------------
        out_overlay = OUT_DIR / "overlays" / f"{group}_{img_id:06d}.jpg"
        out_cam = OUT_DIR / "cams" / f"{group}_{img_id:06d}.npy"
        cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        np.save(str(out_cam), cam)

        rows.append({
            "image_id": img_id,
            "file_name": fn,
            "n_gt_boxes": int(gtb.shape[0]),
            "group": group,
            "gt_smoke": gt_smoke[img_id],
            "pred_smoke": pred_smoke[img_id],
            "target_src": target_src,
            "target_box_x1": float(target_box[0]) if target_box is not None else np.nan,
            "target_box_y1": float(target_box[1]) if target_box is not None else np.nan,
            "target_box_x2": float(target_box[2]) if target_box is not None else np.nan,
            "target_box_y2": float(target_box[3]) if target_box is not None else np.nan,
            "target_best_gt_idx": target_best_gt_idx,
            "target_best_gt_iou": target_best_gt_iou,
            "cam_score": float(score_logit) if not np.isnan(score_logit) else np.nan,
            "cam_iou_top_pct": float(cam_iou) if cam_iou == cam_iou else np.nan,
            "cam_pointing_game": float(cam_pg) if cam_pg == cam_pg else np.nan,
            "cam_energy_inside": float(cam_ein) if cam_ein == cam_ein else np.nan,
        })

        processed += 1
        if processed % 10 == 0 or processed == len(sampled_ids):
            print(f"[XAI] processed {processed}/{len(sampled_ids)}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "xai_pyro_clean_rcnn_gradcpp.csv"
    df.to_csv(out_csv, index=False)
    print("[INFO] Saved:", out_csv)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
