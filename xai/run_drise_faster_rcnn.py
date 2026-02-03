"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions in the CONFIG section below
(e.g., IMG_DIR, COCO_JSON, BASE_OUT_DIR) to match your local directory structure.
"""

import random
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
import torch
import pandas as pd
import json
from utils.rcnn_model import XaiTrainConfig, get_fasterrcnn_model_xai

# -------------------------------------------------------------------
# Reuse utilities from Grad-CAM++ Faster R-CNN script
# -------------------------------------------------------------------
from run_gradcampp_faster_rcnn import (
    load_coco_gt,
    cam_threshold_top_pct,
    iou_mask_boxes,
    pointing_game,
    energy_inside,
    load_predictions,
    index_preds_by_image_id,
)

# -------
# CONFIG
# -------
IMG_DIR   = Path("/path/to/pyro-sdis/images")

COCO_JSON = Path("/path/to/annotations/reference_selected_subset.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_CLASS = 1
NUM_CLASSES = 2

CONF_THRESH = 0.30

# D-RISE params
N_SAMPLES   = 800
MASK_SIZE   = (8, 8)
P_KEEP      = 0.5
RANDOM_SEED = 42

CONF_THRESH_PROP = 0.03

MASTER_SAMPLE_JSON = Path(
    "/path/to/xai_cross_domain_master_sample.json"
)

BASE_PRED_DIR = Path(
    "/path/to/prediction_cache/figlib_bbox"
)

BASE_WEIGHTS_DIR = Path(
    "/path/to/trained_models/faster_rcnn"
)

BASE_OUT_DIR = Path(
    "/path/to/output/xai_cross_domain_results"
)


def load_frcnn_model(weights_path: Path) -> torch.nn.Module:
    cfg = XaiTrainConfig(
        pretrained=False,      # backbone weights are already in the checkpoint
        num_classes=NUM_CLASSES,
        label_smoothing=0.1,
    )
    model = get_fasterrcnn_model_xai(cfg)

    ckpt = torch.load(str(weights_path), map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()
    return model


def generate_random_masks(h: int, w: int, n: int, mask_size=(16, 16), p_keep: float = 0.5) -> np.ndarray:
    mh, mw = mask_size
    cell_h = int(np.ceil(h / mh))
    cell_w = int(np.ceil(w / mw))
    up_h = (mh + 1) * cell_h
    up_w = (mw + 1) * cell_w
    masks_lr = np.random.binomial(1, p_keep, size=(n, mh, mw)).astype(np.float32)
    masks = np.empty((n, h, w), dtype=np.float32)
    for i in range(n):
        m = cv2.resize(masks_lr[i], (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        y0 = np.random.randint(0, up_h - h + 1)
        x0 = np.random.randint(0, up_w - w + 1)
        m = m[y0:y0 + h, x0:x0 + w]
        masks[i] = np.clip(m, 0.0, 1.0)
    return masks


def drise_similarity(
        target_box: np.ndarray,
        target_score: float,
        target_label: int,
        prop_box: np.ndarray,
        prop_score: float,
        prop_label: int,
) -> float:
    xA = max(float(target_box[0]), float(prop_box[0]))
    yA = max(float(target_box[1]), float(prop_box[1]))
    xB = min(float(target_box[2]), float(prop_box[2]))
    yB = min(float(target_box[3]), float(prop_box[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)

    area_t = max(0.0, float(target_box[2] - target_box[0])) * max(0.0, float(target_box[3] - target_box[1]))
    area_p = max(0.0, float(prop_box[2] - prop_box[0])) * max(0.0, float(prop_box[3] - prop_box[1]))
    union = area_t + area_p - inter + 1e-9

    iou = inter / union
    cls_sim = 1.0 if int(target_label) == int(prop_label) else 0.0
    score_sim = float(target_score) * float(prop_score)
    return iou * cls_sim * score_sim


def log_issue(log_rows, img_id, file_name, reason, group=None):
    log_rows.append(
        {"image_id": img_id, "file_name": file_name, "group": group, "reason": reason}
    )


@torch.inference_mode()
def frcnn_predict_single(model: torch.nn.Module, img_np: np.ndarray):
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().to(DEVICE)
    outputs = model([img_t])[0]
    boxes = outputs["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = outputs["scores"].detach().cpu().numpy().astype(np.float32)
    labels = outputs["labels"].detach().cpu().numpy().astype(np.int64)
    return boxes, scores, labels


@torch.inference_mode()
def drise_for_target(
        model: torch.nn.Module,
        orig_rgb: np.ndarray,
        target_box: np.ndarray,
        target_score: float,
        target_label: int,
        n_samples: int = N_SAMPLES,
        mask_size: Tuple[int, int] = MASK_SIZE,
        p_keep: float = P_KEEP,
) -> np.ndarray:
    h, w = orig_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    masks = generate_random_masks(h, w, n_samples, mask_size=mask_size, p_keep=p_keep)

    saliency = np.zeros((h, w), dtype=np.float32)
    norm = 0.0

    base = orig_rgb.astype(np.float32) / 255.0
    baseline = cv2.GaussianBlur(base, (0, 0), 15)

    for i in range(n_samples):
        m = masks[i]
        masked = base * m[..., None] + baseline * (1.0 - m[..., None])

        boxes_p, scores_p, labels_p = frcnn_predict_single(model, masked)

        sim_best = 0.0
        for b, s, lab in zip(boxes_p, scores_p, labels_p):
            if s < CONF_THRESH_PROP:
                continue
            sim = drise_similarity(target_box, target_score, target_label, b, s, lab)
            if sim > sim_best:
                sim_best = sim

        if sim_best > 0:
            saliency += sim_best * m
            norm += sim_best

    if norm <= 1e-12 or saliency.max() <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)

    saliency /= norm
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-6)
    low, high = np.percentile(saliency, [2, 98])
    saliency = np.clip((saliency - low) / (high - low + 1e-6), 0.0, 1.0)
    saliency = saliency ** 1.5
    saliency = saliency / (saliency.max() + 1e-6)
    return saliency


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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
            xA = max(a[0], b[0])
            yA = max(a[1], b[1])
            xB = min(a[2], b[2])
            yB = min(a[3], b[3])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            return inter / (areaA + areaB - inter + 1e-9)

        ious = [iou_xyxy(b, gt_big) for b in boxes]
        return int(np.argmax(ious))

    return int(np.argmax(scores)) if boxes.shape[0] > 0 else -1


def load_master_samples(json_path: Path) -> Dict[int, Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Master sample JSON must be a list of dicts.")
    by_id: Dict[int, Dict] = {}
    for d in data:
        img_id = int(d["image_id"])
        by_id[img_id] = {
            "file_name": d.get("file_name"),
            "n_gt_boxes": int(d.get("n_gt_boxes", 0)),
            "has_smoke": int(d.get("has_smoke", 0)),
            "pred_smoke_ref": int(d.get("pred_smoke_ref", 0)),
        }
    return by_id


def overlay_like_gradcam(orig_rgb: np.ndarray, cam: np.ndarray,
                         alpha_max: float = 0.5,
                         alpha_min: float = 0.10) -> np.ndarray:
    cam = np.clip(cam.astype(np.float32), 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    a = (alpha_min + cam * (alpha_max - alpha_min)).astype(np.float32)[..., None]
    overlay = orig_rgb.astype(np.float32) * (1.0 - a) + heatmap.astype(np.float32) * a
    return overlay.astype(np.uint8)


def run_drise_for_config(config_name: str):
    print(f"\n==============================")
    print(f"[CONFIG] {config_name}")
    print(f"==============================")

    # Build paths for this config
    predictions_pkl = BASE_PRED_DIR / config_name / "predictions.pkl"
    rcnn_weights    = BASE_WEIGHTS_DIR / config_name / "fasterrcnn_best.pth"
    out_dir         = BASE_OUT_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "drise_missing_predictions_log.csv"

    # Set up subdirs
    overlays_dir = out_dir / "overlays"
    cams_dir     = out_dir / "cams"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    id2file, gt_boxes = load_coco_gt(COCO_JSON)
    print(f"[INFO] COCO images in JSON: {len(id2file)}")

    model = load_frcnn_model(rcnn_weights)

    preds = load_predictions(predictions_pkl)
    preds_by_id = index_preds_by_image_id(preds)
    print(f"[INFO] predictions.pkl entries: {len(preds)}")
    print(f"[INFO] unique image_ids in predictions.pkl: {len(preds_by_id)}")

    master_samples = load_master_samples(MASTER_SAMPLE_JSON)
    master_ids = sorted(master_samples.keys())
    print(f"[INFO] master_samples entries: {len(master_ids)}")

    sampled_ids = [i for i in master_ids if i in id2file]
    print(f"[INFO] sampled_ids intersecting COCO: {len(sampled_ids)}")

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
    log_rows = []

    for k, img_id in enumerate(sampled_ids, start=1):
        fn = id2file[img_id]
        img_path = IMG_DIR / fn
        rgb = read_rgb(img_path)

        group = group_of[img_id]
        gtb = gt_boxes[img_id]
        p = preds_by_id.get(img_id, None)

        if p is None:
            log_issue(log_rows, img_id, fn, "missing_prediction_entry", group)
            rows.append({
                "image_id": img_id,
                "file_name": fn,
                "group": group,
                "gt_smoke": gt_smoke[img_id],
                "pred_smoke": pred_smoke[img_id],
                "target_src": "none",
                "target_box_x1": np.nan,
                "target_box_y1": np.nan,
                "target_box_x2": np.nan,
                "target_box_y2": np.nan,
                "target_best_gt_idx": -1,
                "target_best_gt_iou": np.nan,
                "drise_score": np.nan,
                "cam_iou_top_pct": np.nan,
                "cam_pointing_game": np.nan,
                "cam_energy_inside": np.nan,
            })
            continue

        boxes = np.asarray(p.get("boxes", []), dtype=np.float32)
        scores = np.asarray(p.get("scores", []), dtype=np.float32)
        labels = np.asarray(p.get("labels", []), dtype=np.int64)

        if boxes.shape[0] == 0 or scores.shape[0] == 0:
            log_issue(log_rows, img_id, fn, "empty_boxes_in_cache", group)
            rows.append({
                "image_id": img_id,
                "file_name": fn,
                "group": group,
                "gt_smoke": gt_smoke[img_id],
                "pred_smoke": pred_smoke[img_id],
                "target_src": "none",
                "target_box_x1": np.nan,
                "target_box_y1": np.nan,
                "target_box_x2": np.nan,
                "target_box_y2": np.nan,
                "target_best_gt_idx": -1,
                "target_best_gt_iou": np.nan,
                "drise_score": np.nan,
                "cam_iou_top_pct": np.nan,
                "cam_pointing_game": np.nan,
                "cam_energy_inside": np.nan,
            })
            continue

        idx = choose_target_index_for_group(group, boxes, scores, labels, gtb)
        if idx < 0:
            log_issue(log_rows, img_id, fn, "no_usable_prediction_for_target", group)
            rows.append({
                "image_id": img_id,
                "file_name": fn,
                "group": group,
                "gt_smoke": gt_smoke[img_id],
                "pred_smoke": pred_smoke[img_id],
                "target_src": "none",
                "target_box_x1": np.nan,
                "target_box_y1": np.nan,
                "target_box_x2": np.nan,
                "target_box_y2": np.nan,
                "target_best_gt_idx": -1,
                "target_best_gt_iou": np.nan,
                "drise_score": np.nan,
                "cam_iou_top_pct": np.nan,
                "cam_pointing_game": np.nan,
                "cam_energy_inside": np.nan,
            })
            continue

        target_box = boxes[idx]
        target_score = float(scores[idx])
        target_label = int(labels[idx])

        if gtb.shape[0] > 0:
            def iou_xyxy(a, b):
                xA = max(a[0], b[0])
                yA = max(a[1], b[1])
                xB = min(a[2], b[2])
                yB = min(a[3], b[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                return inter / (areaA + areaB - inter + 1e-9)

            ious_to_gt = np.array([iou_xyxy(target_box, g) for g in gtb], dtype=np.float32)
            best_gt_idx = int(np.argmax(ious_to_gt))
            best_gt_iou = float(ious_to_gt[best_gt_idx])
        else:
            best_gt_idx = -1
            best_gt_iou = np.nan

        cam = drise_for_target(
            model=model,
            orig_rgb=rgb,
            target_box=target_box,
            target_score=target_score,
            target_label=target_label,
            n_samples=N_SAMPLES,
            mask_size=MASK_SIZE,
            p_keep=P_KEEP,
        )

        overlay = overlay_like_gradcam(rgb, cam, alpha_max=0.5)

        out_name = f"{group}_{img_id:06d}.jpg"
        cv2.imwrite(str(overlays_dir / out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        np.save(str(cams_dir / out_name.replace(".jpg", ".npy")), cam.astype(np.float32))

        if gtb.shape[0] > 0:
            mask_top = cam_threshold_top_pct(cam, 0.15)
            cam_iou = iou_mask_boxes(mask_top, gtb)
            cam_pg = pointing_game(cam, gtb)
            cam_ein = energy_inside(cam, gtb)
        else:
            cam_iou = cam_pg = cam_ein = np.nan

        rows.append({
            "image_id": img_id,
            "file_name": fn,
            "group": group,
            "gt_smoke": gt_smoke[img_id],
            "pred_smoke": pred_smoke[img_id],
            "target_src": "pred_box",
            "target_box_x1": float(target_box[0]),
            "target_box_y1": float(target_box[1]),
            "target_box_x2": float(target_box[2]),
            "target_box_y2": float(target_box[3]),
            "target_best_gt_idx": best_gt_idx,
            "target_best_gt_iou": best_gt_iou,
            "drise_score": target_score,
            "cam_iou_top_pct": float(cam_iou) if cam_iou == cam_iou else np.nan,
            "cam_pointing_game": float(cam_pg) if cam_pg == cam_pg else np.nan,
            "cam_energy_inside": float(cam_ein) if cam_ein == cam_ein else np.nan,
        })

        if k % 10 == 0 or k == len(sampled_ids):
            print(f"[D-RISE FRCNN {config_name}] processed {k}/{len(sampled_ids)}")

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"xai_pyro_clean_{config_name}_drise.csv"
    df.to_csv(out_csv, index=False)
    print("[INFO] Saved:", out_csv)

    if log_rows:
        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(log_csv, index=False)
        print(f"[INFO] Logged issues to: {log_csv}")
    else:
        print("[INFO] No missing/invalid predictions detected.")


def main():
    configs: List[str] = [
        #"rcnn_aug_nows_lr00025_fold3_seed_500",
        #"rcnn_aug_ws_dynamic_lr00025_seed_22",
        "rcnn_aug_ws_static_lr0005_seed_22",
    ]
    for cfg in configs:
        run_drise_for_config(cfg)


if __name__ == "__main__":
    main()
