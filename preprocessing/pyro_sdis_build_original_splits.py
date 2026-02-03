"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions below
(e.g., json_path, images_root) to match your local directory structure.
"""

from __future__ import annotations
import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image

IMAGE_DIR = "/path/to/pyro-sdis/images"
LABEL_DIR = "/path/to/pyro-sdis/labels"
OUTPUT_DIR = "/path/to/pyro-sdis/splits_test"


def parse_event_id(filename: str) -> str:

    parts = filename.split("_")
    return parts[0] + "_" + parts[1]


def collect_event_stats(img_dir: str, lbl_dir: str) -> Dict[str, Dict]:

    events: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"smoke": 0, "nosmoke": 0, "images": []}
    )

    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"):
            continue

        event_id = parse_event_id(fname)
        label_file = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))

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
        event_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
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
            "test": event_ratios[2] * total_imgs,
        }

        counts = {
            split: sum(len(events[e]["images"]) for e in eids)
            for split, eids in assignments.items()
        }
        img_score = sum(
            abs(counts[s] - tgt[s]) / max(tgt[s], 1e-8)
            for s in ("train", "val", "test")
        )

        ratios = {}
        for split, eids in assignments.items():
            smoke = sum(events[e]["smoke"] for e in eids)
            total = sum(
                events[e]["smoke"] + events[e]["nosmoke"] for e in eids
            )
            ratios[split] = smoke / max(1, total)

        avg_ratio = sum(ratios.values()) / 3.0
        imbalance_score = sum(abs(ratios[s] - avg_ratio) for s in ratios)

        return img_score + imbalance_score

    def try_split(ratios: Tuple[float, float, float]) -> Dict[str, List[str]]:

        best_assign: Dict[str, List[str]] = {}
        best_score = float("inf")

        for _ in range(2000):
            random.shuffle(event_ids)
            n_train = round(ratios[0] * n_events)
            n_val = round(ratios[1] * n_events)
            n_test = n_events - n_train - n_val

            assign = {
                "train": event_ids[:n_train],
                "val": event_ids[n_train : n_train + n_val],
                "test": event_ids[n_train + n_val :],
            }
            sc = score_split(assign)
            if sc < best_score:
                best_score = sc
                best_assign = assign

        return best_assign

    return try_split(event_ratios)


def write_yolo_files(
        events: Dict[str, Dict],
        splits: Dict[str, List[str]],
        out_dir: str,
        img_dir: str,
        lbl_dir: str,
        copy_files: bool = False,
) -> None:

    os.makedirs(out_dir, exist_ok=True)

    for split, eids in splits.items():
        list_path = os.path.join(out_dir, f"{split}.txt")
        with open(list_path, "w") as f:
            for eid in eids:
                for fname in events[eid]["images"]:
                    f.write(os.path.join(img_dir, fname) + "\n")

    if not copy_files:
        return

    for split, eids in splits.items():
        img_out = os.path.join(out_dir, split, "images")
        lbl_out = os.path.join(out_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for eid in eids:
            for fname in events[eid]["images"]:
                src_img = os.path.join(img_dir, fname)
                src_lbl = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
                shutil.copy(src_img, img_out)
                shutil.copy(src_lbl, lbl_out)


def write_yolo_data_yaml(out_dir: str, img_dir: str) -> None:

    yaml_content = (
        "# YOLOv11 dataset config\n"
        f"path: {os.path.abspath(img_dir)}\n"
        f"train: {os.path.join(out_dir, 'train.txt')}\n"
        f"val: {os.path.join(out_dir, 'val.txt')}\n"
        f"test: {os.path.join(out_dir, 'test.txt')}\n\n"
        "names:\n"
        "  0: smoke\n"
    )

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)


def write_coco_json(
        events: Dict[str, Dict],
        split_ids: List[str],
        split_name: str,
        out_dir: str,
        img_dir: str,
        lbl_dir: str,
) -> None:

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for eid in split_ids:
        for fname in events[eid]["images"]:
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))

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

            with open(lbl_path, "r") as f:
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

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{split_name}.json")
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)


def main() -> None:

    copy_files = False
    export_coco = True
    export_yaml = True

    events = collect_event_stats(IMAGE_DIR, LABEL_DIR)
    splits = hybrid_event_split(events, event_ratios=(0.6, 0.2, 0.2))

    write_yolo_files(
        events=events,
        splits=splits,
        out_dir=OUTPUT_DIR,
        img_dir=IMAGE_DIR,
        lbl_dir=LABEL_DIR,
        copy_files=copy_files,
    )

    if export_yaml:
        write_yolo_data_yaml(OUTPUT_DIR, IMAGE_DIR)

    if export_coco:
        for split_name in ("train", "val", "test"):
            write_coco_json(
                events=events,
                split_ids=splits[split_name],
                split_name=split_name,
                out_dir=OUTPUT_DIR,
                img_dir=IMAGE_DIR,
                lbl_dir=LABEL_DIR,
            )

    print("=== Final 60/20/20 Split Summary ===")
    for split_name, eids in splits.items():
        smoke = sum(events[e]["smoke"] for e in eids)
        nosmoke = sum(events[e]["nosmoke"] for e in eids)
        n_imgs = smoke + nosmoke
        ratio = smoke / max(1, n_imgs)
        print(
            f"{split_name.upper()}: {len(eids)} events, {n_imgs} images "
            f"(smoke={smoke}, nosmoke={nosmoke}, smoke_ratio={ratio:.2f})"
        )

if __name__ == "__main__":
    main()
