import torch
import json, os, random
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use("Agg")  # use non-GUI backend for plotting
from torchvision.ops import box_iou
import cv2
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

try:
    from tqdm.auto import tqdm as tqdm_nb
    from tqdm import tqdm as tqdm_cli
except Exception:
    tqdm_nb = tqdm_cli = None

def get_tqdm(notebook_mode: bool):
    """Pick tqdm variant for console vs. notebook. Returns identity if tqdm unavailable."""
    if tqdm_cli is None:
        return lambda x, **kw: x
    return tqdm_nb if notebook_mode else tqdm_cli

def log_class_balance_rcnn(epoch, sampler_weights, sampler_labels, out_dir, cfg):
    """Logs per-epoch class balance during dynamic weighted sampling."""
    try:
        sampled_indices = torch.multinomial(
            torch.as_tensor(sampler_weights, dtype=torch.float32),
            num_samples=len(sampler_weights),
            replacement=True
        )
        labels_t = torch.as_tensor(sampler_labels, dtype=torch.long)
        sampled_labels = labels_t.index_select(0, sampled_indices)
        counts = torch.bincount(sampled_labels, minlength=2)
        total = int(counts.sum())

        dist = {
            "epoch": int(epoch + 1),
            "smoke_samples": int(counts[1].item()),
            "background_samples": int(counts[0].item()),
            "ratio_smoke": round(float(counts[1]) / total, 4),
            "ratio_background": round(float(counts[0]) / total, 4),
            "total_samples": total,
        }

        log_path = out_dir / "class_balance_log.json"
        if log_path.exists():
            prev = json.loads(log_path.read_text())
        else:
            prev = [{
                "run_name": getattr(cfg, "run_name", "rcnn_dynamic_run"),
                "project": str(out_dir.parent),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epochs_total": int(cfg.epochs),
                "batch_size": int(cfg.batch_size),
                "sampling_mode": getattr(cfg, "sampling_mode", "Dynamic WeightedRandomSampler"),
                "records": []
            }]

        prev[0]["records"].append(dist)
        log_path.write_text(json.dumps(prev, indent=2))

        print(f"[LOG] Epoch {epoch+1}: "
              f"Smoke {dist['ratio_smoke']*100:.1f}% | "
              f"Background {dist['ratio_background']*100:.1f}% "
              f"({dist['smoke_samples']} vs {dist['background_samples']})")
    except Exception as e:
        print(f"[WARN] Could not write class_balance_log.json: {type(e).__name__}: {e}")



def rescale_boxes_to_original(boxes: np.ndarray, orig_size, scaled_size) -> np.ndarray:
    """Rescale [x1,y1,x2,y2] boxes from scaled_size back to orig_size."""
    if boxes.shape[0] == 0:
        return boxes
    orig_h, orig_w = orig_size
    new_h, new_w = scaled_size
    sx = orig_w / new_w
    sy = orig_h / new_h
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= sx  # scale x
    boxes[:, [1, 3]] *= sy  # scale y
    return boxes


def debug_gt_vs_preds(model, data_loader, device, out_dir=None, num_images=5, score_thr=0.3):
    """
    Save debug images with GT (green) and predictions (red).
    Also prints IoU and label sanity info to console.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    images, targets = next(iter(data_loader))   # take first batch
    images = [img.to(device) for img in images]

    with torch.no_grad():
        outputs = model(images)

    for i in range(min(num_images, len(images))):
        img = images[i].permute(1,2,0).cpu().numpy()
        gt_boxes = targets[i]["boxes"].cpu()
        gt_labels = targets[i]["labels"].cpu()

        pred_boxes = outputs[i]["boxes"].cpu()
        pred_scores = outputs[i]["scores"].cpu()
        pred_labels = outputs[i]["labels"].cpu()

        # Filter predictions by score threshold
        keep = pred_scores > score_thr
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # --- Label sanity check ---
        print(f"\n[Image {i}]")
        print(f" GT: {len(gt_boxes)} boxes | Pred: {len(pred_boxes)} boxes (thr={score_thr})")
        if len(gt_labels) > 0:
            print(f" GT labels range: min={gt_labels.min().item()}, max={gt_labels.max().item()}")
        else:
            print(" GT labels: none")
        if len(pred_labels) > 0:
            print(f" Pred labels range: min={pred_labels.min().item()}, max={pred_labels.max().item()}")
        else:
            print(" Pred labels: none")

        # IoU matrix
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            print(" IoU matrix:")
            print(ious.numpy())
        else:
            print(" No IoU to compute (empty GT or predictions).")

        # --- Visualization ---
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img)

        # GT boxes (green)
        for box, lbl in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="g", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1, f"GT {lbl.item()}", color="g", fontsize=8)

        # Predicted boxes (red)
        for box, score, lbl in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1, f"P {lbl.item()} {score:.2f}", color="r", fontsize=8)

        # Save image
        out_path = Path(out_dir) / f"debug_img_{i}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f" Saved → {out_path}")


def visualize_val_predictions(model, dataset, device=None, out_dir=None, num_images=20, score_thresh=0.3):
    """
    Show multiple validation samples with GT (red) and predictions (green).
    Args:
        model: trained Faster R-CNN model
        dataset: val dataset (CocoDetDataset)
        device: "cuda" or "cpu"
        num_images: how many random images to show
        score_thresh: filter predictions by confidence
        out_dir: out folder
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

    for idx in indices:
        img, tgt = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]
            # --- DEBUG: print size info ---
        orig_size = tgt["orig_size"].tolist()
        scaled_size = tgt["scaled_size"].tolist()
        #print(f"[DEBUG] Img {tgt['image_id'].item()} → scaled_size={scaled_size}, orig_size={orig_size}")
        #print(f"[DEBUG] First raw box: {pred['boxes'][0].cpu().numpy() if len(pred['boxes'])>0 else 'None'}")


        pred_boxes = pred["boxes"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        # Denormalize image
        img_np = denormalize(img)  # consistent with the helper above

        # Draw GT (red)
        for (x1, y1, x2, y2) in tgt["boxes"].cpu().numpy():
            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Draw predictions (green)
        for (box, score) in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, f"{score:.2f}", (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show figure
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.title(f"Img {tgt['image_id'].item()} | GT=red, Pred=green (thr={score_thresh})")
        plt.axis("off")
        save_path = os.path.join(out_dir, f"val_pred_{idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()   # prevent figures from staying open


def visualize_augmentations_before_after(dataset, out_dir: Path, n=6):
    """
    Save a grid of original vs augmented samples for visual verification.
    Each row: [original | augmented].
    Bounding boxes shown in green.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aug_preview_before_after.jpg"

    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for row, idx in enumerate(idxs):
        try:
            # --- 1) Load original image ---
            img_info = dataset.coco.loadImgs(dataset.ids[idx])[0]
            file_name = img_info["file_name"].replace("\\", "/")
            if file_name.startswith("train/"):
                file_name = file_name[len("train/"):]
            orig_path = Path(dataset.images_dir) / file_name
            orig_img = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
            if orig_img is None:
                raise FileNotFoundError(orig_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            ann_ids = dataset.coco.getAnnIds(imgIds=[dataset.ids[idx]])
            anns = dataset.coco.loadAnns(ann_ids)
            boxes_xyxy = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                boxes_xyxy.append([x, y, x + w, y + h])

            # --- 2) Apply augmentation pipeline manually ---
            transformed = dataset.transforms(
                image=orig_img,
                bboxes=boxes_xyxy,
                labels=[1] * len(boxes_xyxy)
            )
            aug_img = transformed["image"]
            aug_boxes = transformed["bboxes"]

            # --- 3) Normalize dtype for consistent plotting ---
            if torch.is_tensor(aug_img):
                aug_img = aug_img.detach().cpu().permute(1, 2, 0).numpy()

            # Ensure correct range
            if aug_img.max() <= 1.0:
                aug_img = (aug_img * 255.0).astype(np.uint8)
            else:
                aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

            orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)

            # Optional debug (helps diagnose future color issues)
            # print(f"[DEBUG] idx={idx} aug dtype={aug_img.dtype}, min={aug_img.min()}, max={aug_img.max()}")

            # --- 4) Plot original and augmented side by side ---
            for col, (img, boxes, title) in enumerate([
                (orig_img, boxes_xyxy, "Original"),
                (aug_img, aug_boxes, "Augmented"),
            ]):
                ax = axes[row][col] if n > 1 else axes[col]
                ax.imshow(img)
                for (x1, y1, x2, y2) in boxes:
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=1.5, edgecolor="lime", facecolor="none"
                    )
                    ax.add_patch(rect)
                ax.set_title(f"{title} — {len(boxes)} boxes")
                ax.axis("off")

        except Exception as e:
            print(f"[WARN] Could not visualize idx {idx}: {e}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved augmentation before/after preview → {out_path}")





def denormalize(img_tensor):
    """
    Your pipeline feeds [0,1] tensors to the model (no external normalization),
    so for visualization just scale to uint8.
    """
    img = img_tensor.clone().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).clip(0, 255).astype("uint8")
    return img


def visualize_box_drops(dataset, num_samples=5, mode="any", out_dir="viz_drops"):
    """
    Visualize random samples where boxes were dropped due to augmentation.

    Args:
        dataset: CocoDetDataset
        num_samples: how many images to save
        mode: "full", "partial", or "any"
        out_dir: directory to save the images
    """
    os.makedirs(out_dir, exist_ok=True)

    shown, tries = 0, 0
    while shown < num_samples and tries < len(dataset) * 5:
        tries += 1
        img, target = dataset[random.randint(0, len(dataset)-1)]

        orig_count = int(target.get("orig_count", 0))
        final_count = int(target.get("final_count", len(target["boxes"])))

        if mode == "full" and not (orig_count > 0 and final_count == 0):
            continue
        if mode == "partial" and not (orig_count > final_count > 0):
            continue
        if mode == "any" and not (orig_count > final_count):
            continue

        # --- denormalize if needed ---
        if torch.is_tensor(img):
            img_np = denormalize(img)
        else:
            img_np = img.copy()

        img_vis = img_np.copy()

        # red = original boxes
        if "orig_boxes" in target:
            for (x1, y1, x2, y2) in target["orig_boxes"]:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # green = surviving boxes
        for (x1, y1, x2, y2) in target["boxes"]:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # save figure
        img_id = target["image_id"].item()
        out_path = os.path.join(out_dir, f"viz_{img_id}_orig{orig_count}_final{final_count}.png")

        plt.figure(figsize=(6, 6))
        plt.imshow(img_vis)
        plt.title(f"Image {img_id} | orig={orig_count}, final={final_count}\nRed=original, Green=after aug")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Saved visualization → {out_path}")
        shown += 1

def plot_training_curves(csv_path: Path, out_dir: Path) -> None:
    """
    Plot training and validation curves from the training_log.csv using seaborn.
    Figures are saved into a dictionary.
    """

    sns.set_theme(style="whitegrid")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Handle NaNs
    df = df.replace({np.nan: None})

    figs = {}

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    loss_cols = ["train_loss_classifier", "train_loss_box_reg",
                 "train_loss_objectness", "train_loss_rpn_box_reg",
                 "val_loss_classifier", "val_loss_box_reg",
                 "val_loss_objectness", "val_loss_rpn_box_reg"]
    for col in loss_cols:
        if col in df:
            sns.lineplot(data=df, x="epoch", y=col, label=col, ax=ax)
    ax.set_title("Training & Validation Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    figs["loss_curves"] = fig
    fig.savefig(plots_dir / "loss_curves.png")
    plt.close(fig)

    # --- mAP curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in ["mAP_50", "mAP_50_95", "mAP_75"]:
        if col in df:
            sns.lineplot(data=df, x="epoch", y=col, label=col, marker="o", ax=ax)
    ax.set_title("COCO mAP Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    figs["map_curves"] = fig
    fig.savefig(plots_dir / "map_curves.png")
    plt.close(fig)

    # --- AR curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in ["AR_1", "AR_10", "AR_100"]:
        if col in df:
            sns.lineplot(data=df, x="epoch", y=col, label=col, marker="o", ax=ax)
    ax.set_title("COCO AR Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Recall")
    figs["ar_curves"] = fig
    fig.savefig(plots_dir / "ar_curves.png")
    plt.close(fig)

    # --- Learning rate schedule ---
    if "lr" in df:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x="epoch", y="lr", label="Learning Rate", marker="o", ax=ax)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        figs["lr_curve"] = fig
        fig.savefig(plots_dir / "lr_curve.png")
        plt.close(fig)

    print(f"Saved training curves → {plots_dir}")


def plot_pr_and_confusion(model, data_loader, device, out_dir: Path):
    """
    Generate Precision-Recall curve and confusion matrices for final test set.

    Implementation notes:
    - Binary framing: "smoke present" (≥1 GT box) vs "background"
    - Score = max detection score per image (0.5 threshold used for predicted label)
    - Saves PNGs locally and logs to W&B (if active)
    """
    model.eval()
    y_true, y_scores, y_pred = [], [], []

    with torch.inference_mode():
        for images, targets in get_tqdm(False)(data_loader, desc="Generating PR & Confusion", unit="batch"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                # GT: smoke present if at least 1 box
                y_true.append(1 if len(tgt["labels"]) > 0 else 0)

                if out["scores"].numel() > 0:
                    max_score = out["scores"].max().item()
                    y_scores.append(max_score)
                    y_pred.append(1 if max_score > 0.5 else 0)
                else:
                    y_scores.append(0.0)
                    y_pred.append(0)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    figs = {}

    # --- Precision-Recall Curve ---
    if len(set(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
        ap = average_precision_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f"AP={ap:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (Smoke vs Background)")
        ax.legend(); ax.grid(True)
        pr_plot = plots_dir / "pr_curve.png"
        fig.savefig(pr_plot, dpi=200)
        figs["pr_curve"] = fig
        plt.close(fig)

    # --- Confusion Matrices (raw + normalized computed from same cm) ---
    if len(y_true) > 0 and len(y_pred) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Raw
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["background","smoke"],
                    yticklabels=["background","smoke"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        conf = plots_dir / "confusion_matrix.png"
        fig.savefig(conf, dpi=200)
        figs["confusion_matrix"] = fig
        plt.close(fig)

        # Normalized (row-wise)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = cm.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1  # avoid division by zero
            cm_normalized = cm.astype("float") / denom
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=["background", "smoke"],
                    yticklabels=["background", "smoke"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix (per class %)")
        conf_norm = plots_dir / "confusion_matrix_normalized.png"
        fig.savefig(conf_norm, dpi=200)
        figs["confusion_matrix_normalized"] = fig
        plt.close(fig)

    print(f"Saved PR curve and confusion matrix to {plots_dir}")
