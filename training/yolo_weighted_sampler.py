import json
import datetime
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_yolo_dataset

class ResettableDataLoader:
    def __init__(self, dataloader, dataset=None):
        # IMPORTANT: assign internals directly to avoid __getattr__ recursion
        self._dl = dataloader
        self._dataset = dataset

    # ---- iteration/len ----
    def __iter__(self):
        return iter(self._dl)

    def __len__(self):
        return len(self._dl)

    # ---- proxy any missing attribute to the underlying DataLoader ----
    def __getattr__(self, name):
        # Called only if the attribute wasn't found on self
        return getattr(self._dl, name)

    @property
    def dataset(self):
        return getattr(self._dl, "dataset", self._dataset)

    @property
    def sampler(self):
        return getattr(self._dl, "sampler", None)

    @property
    def batch_sampler(self):
        return getattr(self._dl, "batch_sampler", None)

    # ---- the method Ultralytics calls after closing mosaic ----
    def reset(self):
        # No-op; the iterator will be recreated on next __iter__()
        return


def log_class_balance(trainer):
    """Executed at the end of each epoch via Ultralytics callback system."""
    try:
        sampled_indices = torch.multinomial(
            torch.as_tensor(trainer.sample_weights, dtype=torch.float32),
            num_samples=len(trainer.sample_weights),
            replacement=True
        )
        labels_t = torch.as_tensor(trainer.sample_labels, dtype=torch.long)
        sampled_labels = labels_t.index_select(0, sampled_indices)
        counts = torch.bincount(sampled_labels, minlength=2)
        total = int(counts.sum())

        dist = {
            "epoch": int(trainer.epoch + 1),
            "smoke_samples": int(counts[1].item()),
            "background_samples": int(counts[0].item()),
            "ratio_smoke": round(float(counts[1]) / total, 4),
            "ratio_background": round(float(counts[0]) / total, 4),
            "total_samples": total,
        }

        log_path = trainer.save_dir / "class_balance_log.json"
        if log_path.exists():
            prev = json.loads(log_path.read_text())
        else:
            prev = [{
                "run_name": trainer.args.name,
                "project": str(trainer.args.project),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epochs_total": int(trainer.args.epochs),
                "batch_size": int(trainer.args.batch),
                "sampling_mode": "Dynamic WeightedRandomSampler",
                "records": []
            }]

        prev[0]["records"].append(dist)
        log_path.write_text(json.dumps(prev, indent=2))
        print(f"[LOG] Epoch {trainer.epoch+1}: "
              f"Smoke {dist['ratio_smoke']*100:.1f}% | "
              f"Background {dist['ratio_background']*100:.1f}%")
    except Exception as e:
        print(f"[WARN] Could not write class_balance_log.json: {type(e).__name__}: {e}")


class WeightedSamplerTrainer(DetectionTrainer):
    """
    YOLO Trainer subclass that integrates WeightedRandomSampler
    for dynamic class balancing and logs per-epoch distributions,
    losses, and validation metrics.
    """

    # -----------------
    # Dataset builder
    # -----------------
    def build_dataset(self, dataset_path, mode="train", batch_size=16):
        self.args.batch = batch_size
        self.batch_size = batch_size
        gs = max(int(self.model.stride.max() if self.model else 32), 32)
        dataset = build_yolo_dataset(
            self.args,
            dataset_path,
            batch_size,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )
        if not hasattr(dataset, "batch_size") or dataset.batch_size is None:
            dataset.batch_size = batch_size
        return dataset

    # -------------------------------------------------------------
    # Build dataloader with WeightedRandomSampler
    # -------------------------------------------------------------
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode=mode, batch_size=batch_size)

        # ---- Validation/Test: NO weighted sampler, keep true distribution ----
        if mode != "train":
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
            )

        # ---- TRAIN: dynamic balancing ----
        # Build per-image binary labels from dataset.labels
        labels = []
        for lbl in getattr(dataset, "labels", []):
            has_obj = False
            if isinstance(lbl, dict):
                cls = lbl.get("cls", None)
                if cls is not None:
                    if isinstance(cls, torch.Tensor):
                        has_obj = cls.numel() > 0
                    else:
                        has_obj = len(cls) > 0
            elif isinstance(lbl, (list, tuple)):
                has_obj = len(lbl) > 0
            else:
                has_obj = False
            labels.append(1 if has_obj else 0)

        image_labels = torch.as_tensor(labels, dtype=torch.long)

        counts = torch.bincount(image_labels, minlength=2)
        total = counts.sum().clamp_min(1)
        bg_ratio = counts[0].item() / total.item()
        smoke_ratio = counts[1].item() / total.item()

        if (counts == 0).any():
            print("[WARN] One class has zero images in TRAIN split. Disabling dynamic reweighting for stability.")
            class_w = torch.ones(2, dtype=torch.float32)
        else:
            eps = 1e-8

            # Tempered inverse-frequency
            gamma = 0.3
            freq = counts.float() / total
            class_w_temp = (1.0 / freq.clamp_min(eps)).pow(gamma)
            class_w_temp = class_w_temp / class_w_temp.mean()

            # Exact-target weights for desired mix (t = smoke fraction)
            t = 0.5
            w_bg = (1.0 - t) / counts[0].clamp_min(1).float()
            w_sm = t / counts[1].clamp_min(1).float()
            class_w_tgt = torch.tensor([w_bg, w_sm], dtype=torch.float32)
            class_w_tgt = class_w_tgt / class_w_tgt.mean()

            # Blend
            alpha = 0.5
            class_w = (1 - alpha) * class_w_temp + alpha * class_w_tgt

            # Normalize + clamp
            class_w = class_w / class_w.mean()
            class_w = class_w.clamp(0.7, 1.6)
            class_w = class_w / class_w.mean()

            # --- Post-hoc target correction to hit the desired expected mix exactly ---
            N_bg = max(int(counts[0].item()), 1)
            N_sm = max(int(counts[1].item()), 1)
            w_bg_curr = float(class_w[0].item())
            w_sm_curr = float(class_w[1].item())
            r_bg_target = 1.0 - t
            s_bg = (r_bg_target / max(1.0 - r_bg_target, 1e-12)) * (w_sm_curr * N_sm) / max(w_bg_curr * N_bg, 1e-12)

            class_w = class_w.clone()
            class_w[0] = class_w[0] * float(s_bg)

            # Renormalize to mean=1
            class_w = class_w / class_w.mean()

        # Per-sample weights
        weights = class_w.index_select(0, image_labels).cpu()

        # Expected epoch mix
        exp_bg_w = float(class_w[0]) * counts[0].item()
        exp_sm_w = float(class_w[1]) * counts[1].item()
        exp_den = max(exp_bg_w + exp_sm_w, 1e-12)
        exp_bg_ratio = exp_bg_w / exp_den
        exp_sm_ratio = exp_sm_w / exp_den

        # Keep epoch length comparable to dataset size
        epoch_len = len(dataset)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=epoch_len,
            replacement=True,
        )

        print(f"[INFO] Expected epoch mix (post-correct) → Background: {exp_bg_ratio:.3f} | Smoke: {exp_sm_ratio:.3f}")

        # Expose for your callback
        self.sample_labels = image_labels.cpu().tolist()
        self.sample_weights = weights.tolist()

        # ---- Debug messages ----
        print("[INFO] WeightedRandomSampler enabled — dynamically balancing classes each epoch.")
        print(f"[INFO] Class counts: {counts.tolist()} → ClassWeights(blend): {class_w.tolist()}")
        print(f"[INFO] Initial ratio → Background: {bg_ratio:.3f} | Smoke: {smoke_ratio:.3f}")
        print(f"[INFO] Expected epoch mix → Background: {exp_bg_ratio:.3f} | Smoke: {exp_sm_ratio:.3f}")
        print(f"[INFO] Dataset batch_size confirmed as: {dataset.batch_size}")
        print(f"[INFO] Epoch samples (replacement): {epoch_len}")

        dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=True if self.args.workers and self.args.workers > 0 else False,
        )
        return ResettableDataLoader(dl, dataset=dataset)
