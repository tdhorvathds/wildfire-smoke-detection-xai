"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions below
(e.g., test_txt, image_dir) to match your local directory structure.
"""

from __future__ import annotations
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image


def parse_event_id(filename: str) -> str:

    parts = filename.split("_")
    return parts[0] + "_" + parts[1]


def collect_event_stats(
        img_dir: Path,
        lbl_dir: Path,
        exclude_events: Set[str] | None = None,
        exclude_files: Set[str] | None = None,
) -> Dict[str, Dict]:

    exclude_events = exclude_events or set()
    exclude_files = exclude_files or set()

    events: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"smoke": 0, "nosmoke": 0, "images": []}
    )

    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"):
            continue
        if fname in exclude_files:
            continue

        event_id = parse_event_id(fname)
        if event_id in exclude_events:
            continue

        label_file = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(label_file):
            continue

        with open(label_file, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            events[event_id]["nosmoke"] += 1
        else:
            events[event_id]["smoke"] += 1

        events[event_id]["images"].append(fname)

    return events


def hybrid_event_split(
        events: Dict[str, Dict],
        event_ratios: Tuple[float, float] = (0.6, 0.2),
        seed: int = 42,
) -> Dict[str, List[str]]:

    random.seed(seed)
    event_ids = list(events.keys())
    n_events = len(event_ids)

    def score_split(assignments: Dict[str, List[str]]) -> float:

        total_imgs = sum(len(events[e]["images"]) for e in events)
        tgt = {
            "train": event_ratios[0] * total_imgs,
            "val": event_ratios[1] * total_imgs,
        }

        counts = {
            split: sum(len(events[e]["images"]) for e in eids)
            for split, eids in assignments.items()
        }
        img_score = sum(
            abs(counts[s] - tgt[s]) / max(tgt[s], 1e-8)
            for s in ("train", "val")
        )

        ratios = {}
        for split, eids in assignments.items():
            smoke = sum(events[e]["smoke"] for e in eids)
            tot = sum(
                events[e]["smoke"] + events[e]["nosmoke"] for e in eids
            )
            ratios[split] = smoke / max(1, tot)

        avg_ratio = sum(ratios.values()) / 2.0
        imbalance_score = sum(abs(ratios[s] - avg_ratio) for s in ratios)

        return img_score + imbalance_score

    best_assign: Dict[str, List[str]] | None = None
    best_score = float("inf")

    for _ in range(2000):
        random.shuffle(event_ids)
        n_train = round(event_ratios[0] * n_events)
        n_val = n_events - n_train

        assign = {
            "train": event_ids[:n_train],
            "val": event_ids[n_train:],
        }
        sc = score_split(assign)
        if sc < best_score:
            best_score = sc
            best_assign = assign

    return best_assign if best_assign is not None else {"train": [], "val": []}


def write_yolo_files(
        events: Dict[str, Dict],
        splits: Dict[str, List[str]],
        out_dir: Path | str,
        img_dir: Path,
        lbl_dir: Path,
) -> None:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, eids in splits.items():
        txt_path = out_dir / f"{split}.txt"
        with txt_path.open("w") as f:
            for eid in eids:
                for fname in events[eid]["images"]:
                    f.write(str(img_dir / fname) + "\n")


def write_yolo_data_yaml(out_dir: Path, img_dir: Path, test_txt: str) -> None:

    yaml_content = (
        "# Pyro-SDIS Fold Config\n"
        f"path: {os.path.abspath(img_dir)}\n"
        f"train: {os.path.join(out_dir, 'train.txt')}\n"
        f"val: {os.path.join(out_dir, 'val.txt')}\n"
        f"test: {test_txt}\n\n"
        "names:\n"
        "  0: smoke\n"
    )

    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w") as f:
        f.write(yaml_content)


def write_coco_json(
        events: Dict[str, Dict],
        split_ids: List[str],
        split_name: str,
        out_dir: Path | str,
        img_dir: Path,
        lbl_dir: Path,
) -> None:

    out_dir = Path(out_dir)
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for eid in split_ids:
        for fname in events[eid]["images"]:
            img_path = img_dir / fname
            lbl_path = lbl_dir / fname.replace(".jpg", ".txt")

            if not img_path.exists() or not lbl_path.exists():
                continue

            with Image.open(img_path) as img:
                width, height = img.size

            images.append(
                {
                    "id": img_id,
                    "file_name": fname,
                    "height": height,
                    "width": width,
                }
            )

            with lbl_path.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls, x, y, w, h = map(float, parts)
                    x_min = (x - w / 2.0) * width
                    y_min = (y - h / 2.0) * height
                    box_w = w * width
                    box_h = h * height

                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(cls),
                            "bbox": [x_min, y_min, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 0,
                "name": "smoke",
            }
        ],
    }

    json_path = out_dir / f"{split_name}.json"
    with json_path.open("w") as f:
        json.dump(coco, f, indent=2)


def read_test_lists(test_txt: Path) -> Tuple[Set[str], Set[str]]:

    with test_txt.open("r") as f:
        paths = [ln.strip() for ln in f if ln.strip()]

    test_files = {Path(p).name for p in paths}
    test_events = {parse_event_id(fn) for fn in test_files}
    return test_files, test_events


# ------
# Main
# ------

if __name__ == "__main__":

    # Paths are defined relative to the project root and may need adaptation
    # depending on the local directory structure.
    PARENT_DIR = Path.cwd().parent
    IMAGE_DIR = PARENT_DIR.parent / "data/pyro-sdis/images"
    LABEL_DIR = PARENT_DIR.parent / "data/pyro-sdis/labels"
    OUTPUT_DIR = PARENT_DIR / "test/splits_3fold"
    TEST_TXT = PARENT_DIR / "data/pyro-sdis/splits/test.txt"
    TEST_JSON = PARENT_DIR / "data/pyro-sdis/splits/test.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_files, test_events = read_test_lists(TEST_TXT)
    events = collect_event_stats(
        IMAGE_DIR,
        LABEL_DIR,
        exclude_events=test_events,
        exclude_files=test_files,
    )

    print(f"Excluded {len(test_events)} test events, {len(test_files)} test images.")
    print(f"Remaining {len(events)} events for 3-fold resplit.")

    seeds = [22, 45, 118]
    summary: Dict[str, List[Dict]] = {"folds": []}

    for i, seed in enumerate(seeds, start=1):
        print(f"\n=== Generating Fold {i} (seed={seed}) ===")
        splits = hybrid_event_split(events, event_ratios=(0.6, 0.2), seed=seed)

        fold_out = OUTPUT_DIR / f"fold{i}"
        fold_out.mkdir(parents=True, exist_ok=True)

        write_yolo_files(events, splits, fold_out, IMAGE_DIR, LABEL_DIR)
        write_yolo_data_yaml(fold_out, IMAGE_DIR, str(TEST_TXT))

        for split_name in ("train", "val"):
            write_coco_json(
                events,
                splits[split_name],
                split_name,
                fold_out,
                IMAGE_DIR,
                LABEL_DIR,
            )

        shutil.copy(TEST_TXT, fold_out / "test.txt")
        if TEST_JSON.exists():
            shutil.copy(TEST_JSON, fold_out / "test.json")

        fold_stats: Dict[str, object] = {"fold": i, "seed": seed, "splits": {}}
        for split_name, eids in splits.items():
            smoke = sum(events[e]["smoke"] for e in eids)
            nosmoke = sum(events[e]["nosmoke"] for e in eids)
            n_imgs = smoke + nosmoke
            ratio = smoke / max(1, n_imgs)

            print(
                f"{split_name.upper()}: {len(eids)} events, {n_imgs} images "
                f"(smoke={smoke}, nosmoke={nosmoke}, smoke_ratio={ratio:.2f})"
            )

            fold_stats["splits"][split_name] = {
                "num_events": len(eids),
                "num_images": n_imgs,
                "num_smoke": smoke,
                "num_nosmoke": nosmoke,
                "smoke_ratio": round(ratio, 4),
                "event_ids": sorted(eids),
            }

        n_train = fold_stats["splits"]["train"]["num_images"]
        n_val = fold_stats["splits"]["val"]["num_images"]
        fold_stats["train_val_ratio"] = round(n_train / max(1, n_val), 3)

        summary["folds"].append(fold_stats)

    summary_path = OUTPUT_DIR / "split_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n3-fold resplits created. Summary saved to: {summary_path}")
