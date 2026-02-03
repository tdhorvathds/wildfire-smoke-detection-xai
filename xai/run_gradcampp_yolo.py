"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions in the CONFIG section below
(e.g., IMG_DIR, COCO_JSON, BASE_OUT_DIR) to match your local directory structure.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torch.nn as nn

from ultralytics import YOLO
import pandas as pd

# -------------------------
# CONFIG
# -------------------------

IMG_DIR = Path(
    "/path/to/images"
)

COCO_JSON = Path(
    "/path/to/annotations/coco_annotations.json"
)

PREDICTIONS_PKL = Path(
    "/path/to/prediction_cache/yolo_predictions.pkl"
)

YOLO_WEIGHTS = Path(
    "/path/to/trained_models/yolov11s_weights.pt"
)

OUT_DIR = Path(
    "/path/to/output"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

IMGSZ = 640
CONF_THRESH = 0.30
CAM_TOP_PCT = 0.15
SMOKE_CLASS = 1

MASTER_SAMPLE_JSON = Path(
    "/path/to/xai_cross_domain_master_sample.json"
)

# -------------------------
# Utilities
# -------------------------
def load_coco_gt(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    gt = {img_id: [] for img_id in id2file.keys()}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        gt[img_id].append([x, y, x + w, y + h])

    for k in gt:
        gt[k] = np.array(gt[k], dtype=np.float32) if len(gt[k]) else np.zeros((0, 4), dtype=np.float32)

    return id2file, gt


def load_predictions(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        preds = pickle.load(f)
    if not isinstance(preds, list) or not preds:
        raise ValueError(f"Unexpected predictions.pkl format: type={type(preds)}, len={len(preds) if hasattr(preds,'__len__') else 'NA'}")
    return preds


def index_preds_by_image_id(preds):
    by_id = {}
    for p in preds:
        if "image_id" in p and isinstance(p["image_id"], (int, np.integer)):
            by_id[int(p["image_id"])] = p
    return by_id


def has_pred_smoke(p, conf_thresh: float) -> bool:
    scores = np.asarray(p.get("scores", []), dtype=np.float32)
    return bool(scores.size > 0 and (scores >= conf_thresh).any())

def enable_yolo_grads_train_forward(model: torch.nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for p in model.parameters():
        p.requires_grad_(True)

# -------------------------
# Image preprocessing
# -------------------------
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]
    h0, w0 = shape

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h_new, w_new = new_shape
    r = min(h_new / h0, w_new / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))

    dw = w_new - new_unpad[0]
    dh = h_new - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh


def preprocess_for_yolo_cam(img_path: Path, imgsz: int, device: str):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    lb_rgb, r, dw, dh = letterbox(orig_rgb, new_shape=imgsz)
    x = torch.from_numpy(lb_rgb).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)
    return x, orig_rgb, lb_rgb, r, dw, dh


def cam_to_original(cam_lb: np.ndarray, orig_shape, r: float, dw: float, dh: float):
    h0, w0 = orig_shape[:2]
    H, W = cam_lb.shape

    y1 = int(max(0, round(dh)))
    x1 = int(max(0, round(dw)))
    y2 = int(min(H, round(dh + h0 * r)))
    x2 = int(min(W, round(dw + w0 * r)))

    if y2 <= y1 or x2 <= x1:
        return cv2.resize(cam_lb, (w0, h0), interpolation=cv2.INTER_LINEAR)

    cam_crop = cam_lb[y1:y2, x1:x2]
    cam_orig = cv2.resize(cam_crop, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return cam_orig


# -------------------------
# Detection-linked target
# -------------------------
def box_center_to_pred_indices(cx_lb: float, cy_lb: float, imgsz: int = 640):
    strides = [8, 16, 32]
    offsets = [0, 64 * 64, 64 * 64 + 32 * 32]
    idxs = []
    for s, off in zip(strides, offsets):
        g = imgsz // s
        gx = int(np.clip(cx_lb / s, 0, g - 1))
        gy = int(np.clip(cy_lb / s, 0, g - 1))
        idxs.append(off + gy * g + gx)
    return idxs


def prepare_yolo_for_cam(m: torch.nn.Module):
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
    for p in m.parameters():
        p.requires_grad_(True)
    return m


# -------------------------
# XAI metrics
# -------------------------
def cam_threshold_top_pct(cam: np.ndarray, pct: float):
    cam = cam.astype(np.float32)
    thr = np.quantile(cam.reshape(-1), 1.0 - pct)
    mask = (cam >= thr).astype(np.uint8)
    return mask


def iou_mask_boxes(mask: np.ndarray, boxes_xyxy: np.ndarray):
    if boxes_xyxy.size == 0:
        return np.nan
    h, w = mask.shape
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1i = int(np.clip(x1, 0, w - 1))
        x2i = int(np.clip(x2, 0, w))
        y1i = int(np.clip(y1, 0, h - 1))
        y2i = int(np.clip(y2, 0, h))
        if x2i > x1i and y2i > y1i:
            gt_mask[y1i:y2i, x1i:x2i] = 1
    inter = (mask & gt_mask).sum()
    union = (mask | gt_mask).sum()
    return float(inter / union) if union > 0 else np.nan


def pointing_game(cam: np.ndarray, boxes_xyxy: np.ndarray):
    if boxes_xyxy.size == 0:
        return np.nan
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    for (x1, y1, x2, y2) in boxes_xyxy:
        if (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2):
            return 1.0
    return 0.0


def energy_inside(cam: np.ndarray, boxes_xyxy: np.ndarray):
    if boxes_xyxy.size == 0:
        return np.nan
    h, w = cam.shape
    inside = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1i = int(np.clip(x1, 0, w - 1))
        x2i = int(np.clip(x2, 0, w))
        y1i = int(np.clip(y1, 0, h - 1))
        y2i = int(np.clip(y2, 0, h))
        if x2i > x1i and y2i > y1i:
            inside[y1i:y2i, x1i:x2i] = 1
    total = float(cam.sum()) + 1e-8
    return float((cam * inside).sum() / total)


def assert_autograd_ok():
    if not torch.is_grad_enabled():
        raise RuntimeError("Grad is globally disabled.")
    t = torch.tensor(1.0, requires_grad=True)
    y = t * 2.0
    y.backward()


# -------------------------
# Grad-CAM++
# -------------------------
class GradCAM_YOLO:
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None

    def generate_obj_topk(
            self,
            x: torch.Tensor,
            orig_rgb: np.ndarray,
            r: float, dw: float, dh: float,
            cam_to_original_fn,
            target_indices: list[int] | None = None,
            input_size: int = 512,
            topk: int = 200,
            debug: bool = False,
    ):
        enable_yolo_grads_train_forward(self.model)
        self.activations = None
        self.gradients = None

        m = prepare_yolo_for_cam(self.model)

        with torch.enable_grad():
            out = m(x)

        def collect_head_tensors(obj):
            heads = []
            if torch.is_tensor(obj) and obj.dim() == 4:
                heads.append(obj)
            elif isinstance(obj, (list, tuple)):
                for z in obj:
                    if torch.is_tensor(z) and z.dim() == 4:
                        heads.append(z)
                    elif isinstance(z, (list, tuple)):
                        for t in z:
                            if torch.is_tensor(t) and t.dim() == 4:
                                heads.append(t)
            return heads

        head_tensors = collect_head_tensors(out)
        if not head_tensors:
            raise RuntimeError(f"Could not find any (B,C,H,W) head tensors.")

        head_tensors = sorted(head_tensors, key=lambda t: int(t.shape[-1] * t.shape[-2]), reverse=True)
        head = head_tensors[0]

        self.activations = head
        self.gradients = None
        if head.requires_grad:
            head.retain_grad()
            head.register_hook(lambda g: setattr(self, "gradients", g))

        def extract_class_scores(p, obj_thresh=0.10):
            if not torch.is_tensor(p) or p.dim() != 4:
                raise RuntimeError("Unexpected head tensor.")
            obj = p[:, 4:5, :, :].sigmoid()
            cls = p[:, 5:6, :, :].sigmoid()
            score = obj * cls
            mask = (obj >= obj_thresh).float()
            score = score * mask
            flat = score.flatten(1)
            return flat

        cls_flat = extract_class_scores(head)

        if target_indices is not None and len(target_indices) > 0:
            idx = torch.tensor(target_indices, device=cls_flat.device, dtype=torch.long)
            idx = idx.clamp(0, cls_flat.shape[1] - 1)
            vals = cls_flat[:, idx]
            k_local = min(5, vals.shape[1])
            score = vals.topk(k_local, dim=1).values.sum()
        else:
            k = min(topk, cls_flat.shape[1])
            score = cls_flat.topk(k, dim=1).values.sum()

        m.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        A = self.activations
        dA = self.gradients if self.gradients is not None else getattr(A, "grad", None)
        if dA is None:
            raise RuntimeError("No gradient reached head tensor.")

        weights = gradcampp_weights(A, dA)
        cam = (weights * A).sum(dim=1)
        cam = F.relu(cam)[0].detach().float().cpu().numpy()

        cam = cv2.GaussianBlur(cam, (5, 5), 0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)

        low, high = np.percentile(cam, [5, 99])
        cam = np.clip((cam - low) / (high - low + 1e-6), 0, 1)

        cam_lb = cv2.resize(cam, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        cam_orig = cam_to_original_fn(cam_lb, orig_rgb.shape, r, dw, dh)

        cam_orig = cam_orig ** 2.0
        cam_orig = cam_orig / (cam_orig.max() + 1e-6)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_orig), cv2.COLORMAP_JET)
        overlay = (0.5 * heatmap[..., ::-1] + 0.5 * orig_rgb).astype(np.uint8)

        return cam_orig, overlay, float(score.detach().cpu())


def gradcampp_weights(activations: torch.Tensor, gradients: torch.Tensor, eps: float = 1e-6):
    grad2 = gradients ** 2
    grad3 = gradients ** 3
    denom = 2.0 * grad2 + activations * grad3
    denom = torch.where(denom != 0.0, denom, torch.full_like(denom, eps))
    alpha = grad2 / denom
    alpha = F.relu(alpha)
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    return weights


# -------------------------
# Index helpers
# -------------------------
def decode_concat_index(idx: int):
    n64 = 64 * 64
    n32 = 32 * 32
    if idx < n64:
        s = 64
        off = idx
    elif idx < n64 + n32:
        s = 32
        off = idx - n64
    else:
        s = 16
        off = idx - (n64 + n32)
    y = off // s
    x = off % s
    return s, int(y), int(x)


def encode_concat_index(scale: int, y: int, x: int):
    n64 = 64 * 64
    n32 = 32 * 32
    if scale == 64:
        base = 0
        s = 64
    elif scale == 32:
        base = n64
        s = 32
    elif scale == 16:
        base = n64 + n32
        s = 16
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    return base + y * s + x


def expand_indices(center_indices: list[int], radius: int = 2):
    out = set()
    for idx in center_indices:
        scale, cy, cx = decode_concat_index(int(idx))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy = cy + dy
                xx = cx + dx
                if 0 <= yy < scale and 0 <= xx < scale:
                    out.add(encode_concat_index(scale, yy, xx))
    return sorted(out)


def choose_target_index_for_group(
        group: str,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        gt_boxes: np.ndarray,
) -> int:
    if boxes.shape[0] == 0 or scores.shape[0] == 0:
        return -1

    mask_smoke = (labels == SMOKE_CLASS) & (scores >= CONF_THRESH)
    idx_smoke = np.where(mask_smoke)[0]

    if group in ("TP", "FP") and idx_smoke.size > 0:
        return int(idx_smoke[np.argmax(scores[idx_smoke])])

    if gt_boxes.shape[0] > 0 and boxes.shape[0] > 0:
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        gt_big = gt_boxes[np.argmax(areas)].astype(np.float32)

        def iou_xyxy(a, b):
            xA = max(a[0], b[0]); yA = max(a[1], b[1])
            xB = min(a[2], b[2]); yB = min(a[3], b[3])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            return inter / (areaA + areaB - inter + 1e-9)

        ious = [iou_xyxy(b, gt_big) for b in boxes]
        return int(np.argmax(ious))

    return int(np.argmax(scores)) if boxes.shape[0] > 0 else -1

# -------------------------
# Load master sample file
# -------------------------
def load_master_samples(json_path: Path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Master sample JSON must be a list of dicts.")

    # map image_id -> meta from file
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

# -------------------------
# Main
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    id2file, gt_boxes = load_coco_gt(COCO_JSON)
    image_ids = set(id2file.keys())
    print(f"[INFO] COCO images: {len(image_ids)}")

    preds = load_predictions(PREDICTIONS_PKL)
    preds_by_id = index_preds_by_image_id(preds)

    master_samples = load_master_samples(MASTER_SAMPLE_JSON)

    sampled_ids = [i for i in master_samples.keys() if i in image_ids]
    print(f"[INFO] Using {len(sampled_ids)} images from master sample file")

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

    yolo = YOLO(str(YOLO_WEIGHTS))
    yolo_torch = yolo.model.to(device)

    cam_engine = GradCAM_YOLO(yolo_torch, device=device)

    overlays_dir = OUT_DIR / "overlays"
    cams_dir = OUT_DIR / "cams"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for k, img_id in enumerate(sampled_ids, start=1):
        file_name = id2file[img_id]
        img_path = IMG_DIR / file_name

        x, orig_rgb, lb_rgb, r, dw, dh = preprocess_for_yolo_cam(
            img_path, imgsz=IMGSZ, device=device
        )

        p = preds_by_id.get(img_id, None)
        gtb = gt_boxes[img_id]

        target_src = "none"
        target_box = None

        if p is not None:
            boxes = np.asarray(
                p.get("boxes", np.zeros((0, 4), np.float32)),
                dtype=np.float32,
            )
            scores = np.asarray(
                p.get("scores", np.zeros((0,), np.float32)),
                dtype=np.float32,
            )
            labels = np.asarray(
                p.get("labels", np.zeros((0,), np.int64)),
                dtype=np.int64,
            )

            if 0 < boxes.shape[0] == scores.shape[0]:
                idx = choose_target_index_for_group(
                    group_of[img_id], boxes, scores, labels, gtb
                )
                if idx >= 0:
                    target_box = boxes[idx]
                    target_src = "pred_box"


        if target_box is not None:
            x1, y1, x2, y2 = target_box
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
        elif gtb.shape[0] > 0:
            areas = (gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])
            j = int(np.argmax(areas))
            x1, y1, x2, y2 = gtb[j]
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            target_src = "gt_box"
        else:
            h0, w0 = orig_rgb.shape[:2]
            cx, cy = w0 / 2.0, h0 / 2.0
            target_src = "image_center"

        cx_lb = cx * r + dw
        cy_lb = cy * r + dh
        center_indices = box_center_to_pred_indices(cx_lb, cy_lb, imgsz=IMGSZ)
        target_indices = [center_indices[0]]
        target_indices = expand_indices(target_indices, radius=1)

        assert_autograd_ok()
        cam, overlay, cam_score = cam_engine.generate_obj_topk(
            x=x,
            orig_rgb=orig_rgb,
            r=r,
            dw=dw,
            dh=dh,
            cam_to_original_fn=cam_to_original,
            target_indices=target_indices,
            input_size=IMGSZ,
            debug=False,
        )

        if gtb.shape[0] > 0 and target_box is not None:
            def iou_xyxy(a, b):
                xA = max(a[0], b[0])
                yA = max(a[1], b[1])
                xB = min(a[2], b[2])
                yB = min(a[3], b[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                return inter / (areaA + areaB - inter + 1e-9)

            ious_to_gt = np.array(
                [iou_xyxy(target_box, g) for g in gtb],
                dtype=np.float32,
            )
            target_best_gt_idx = int(np.argmax(ious_to_gt))
            target_best_gt_iou = float(ious_to_gt[target_best_gt_idx])
        else:
            target_best_gt_idx = -1
            target_best_gt_iou = np.nan

        cam_iou = np.nan
        cam_pg = np.nan
        cam_ein = np.nan

        if gtb.shape[0] > 0:
            mask = cam_threshold_top_pct(cam, CAM_TOP_PCT)
            cam_iou = iou_mask_boxes(mask, gtb)
            cam_pg = pointing_game(cam, gtb)
            cam_ein = energy_inside(cam, gtb)

        out_name = f"{group_of[img_id]}_{img_id:06d}.jpg"
        cv2.imwrite(
            str(overlays_dir / out_name),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        np.save(
            str(cams_dir / out_name.replace(".jpg", ".npy")),
            cam.astype(np.float32),
        )

        rows.append({
            "image_id": img_id,
            "file_name": file_name,
            "n_gt_boxes": int(gtb.shape[0]),
            "group": group_of[img_id],
            "gt_smoke": gt_smoke[img_id],
            "pred_smoke": pred_smoke[img_id],
            "target_src": target_src,
            "target_box_x1": float(target_box[0]) if target_box is not None else np.nan,
            "target_box_y1": float(target_box[1]) if target_box is not None else np.nan,
            "target_box_x2": float(target_box[2]) if target_box is not None else np.nan,
            "target_box_y2": float(target_box[3]) if target_box is not None else np.nan,
            "target_best_gt_idx": target_best_gt_idx,
            "target_best_gt_iou": target_best_gt_iou,
            "cam_score": float(cam_score) if not np.isnan(cam_score) else np.nan,
            "cam_iou_top_pct": float(cam_iou) if cam_iou == cam_iou else np.nan,
            "cam_pointing_game": float(cam_pg) if cam_pg == cam_pg else np.nan,
            "cam_energy_inside": float(cam_ein) if cam_ein == cam_ein else np.nan,
        })

        if k % 25 == 0 or k == len(sampled_ids):
            print(f"[XAI] processed {k}/{len(sampled_ids)}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "xai_pyro_clean_yolo.csv"
    df.to_csv(out_csv, index=False)
    print("[INFO] Saved:", out_csv)


if __name__ == "__main__":
    main()
