"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions below
(e.g., csv_dir, images_root) to match your local directory structure.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict


def figlib_to_coco(csv_dir: str, images_root: str, output_json: str) -> None:

    csv_dir = Path(csv_dir)
    images_root = Path(images_root)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "no-smoke"},
            {"id": 1, "name": "smoke"},
        ],
    }

    image_id_counter = 1
    annot_id_counter = 1

    image_registry: Dict[str, int] = {}
    image_labels: Dict[str, int] = {}
    bbox_count = defaultdict(int)
    bbox_areas = []

    csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} annotation CSV files.\n")

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        event_name = csv_file.stem  # e.g. "20160619_FIRE_om-e-mobo-c"
        event_folder = images_root / event_name

        if not event_folder.exists():
            print(f"WARNING: Missing event folder {event_name}, skipping.")
            continue

        df = pd.read_csv(csv_file, sep=None, engine="python")

        df.columns = [c.strip() for c in df.columns]

        required_cols = ["MinX", "MinY", "MaxX", "MaxY", "Filename"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in {csv_file} → {required_cols}")

        df["basename"] = df["Filename"].apply(lambda x: Path(str(x)).name)

        annotated_basenames = set(df["basename"].tolist())

        event_images = sorted([p.name for p in event_folder.iterdir() if p.is_file()])

        for img_name in event_images:
            key = f"{event_name}/{img_name}"
            if key not in image_labels:
                image_labels[key] = 1 if img_name in annotated_basenames else 0

        for img_name in event_images:
            key = f"{event_name}/{img_name}"
            if key not in image_registry:
                img_path = event_folder / img_name
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception:
                    width, height = 1920, 1080

                coco["images"].append(
                    {
                        "id": image_id_counter,
                        "file_name": key,  # e.g. "20160619_FIRE_om-e-mobo-c/1466361757_+02400.jpg"
                        "width": width,
                        "height": height,
                    }
                )

                image_registry[key] = image_id_counter
                image_id_counter += 1

        for _, row in df.iterrows():
            basename = row["basename"]
            key = f"{event_name}/{basename}"
            image_id = image_registry.get(key)

            if image_id is None:

                print(f"WARNING: CSV references missing image: {key}")
                continue

            xmin = float(row["MinX"])
            ymin = float(row["MinY"])
            xmax = float(row["MaxX"])
            ymax = float(row["MaxY"])
            w, h = xmax - xmin, ymax - ymin
            area = w * h

            coco["annotations"].append(
                {
                    "id": annot_id_counter,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
            )

            bbox_count[image_id] += 1
            bbox_areas.append(area)
            annot_id_counter += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    # =======================
    # STATISTICS REPORT
    # =======================
    total_images = len(coco["images"])
    total_annotations = len(coco["annotations"])
    total_smoke_images = sum(image_labels.values())
    total_no_smoke_images = len(image_labels) - total_smoke_images

    annot_per_image = np.array(list(bbox_count.values())) if bbox_count else np.array([])
    bbox_areas_np = np.array(bbox_areas) if bbox_areas else np.array([0])

    print("\n" + "=" * 60)
    print("                  FIgLib COCO ANNOTATION STATISTICS")
    print("=" * 60)
    print(f"Total images:                        {total_images}")
    print(f"Smoke images (with bbox):            {total_smoke_images}")
    print(f"No-smoke images (no bbox):           {total_no_smoke_images}")
    print(f"Total annotations:                   {total_annotations}")
    print("-" * 60)

    if len(annot_per_image) > 0:
        print("Bounding Box Statistics:")
        print(f"   • Avg bboxes per smoke image:       {annot_per_image.mean():.2f}")
        print(f"   • Min bbox area:                    {bbox_areas_np.min():.2f}")
        print(f"   • Max bbox area:                    {bbox_areas_np.max():.2f}")
        print(f"   • Mean bbox area:                   {bbox_areas_np.mean():.2f}")
        print("-" * 60)
        print("Distribution of bounding boxes per image:")
        print(f"   • Images with 1 box:                {(annot_per_image == 1).sum()}")
        print(f"   • Images with 2 boxes:              {(annot_per_image == 2).sum()}")
        print(f"   • Images with 3+ boxes:             {(annot_per_image >= 3).sum()}")
    else:
        print("No annotations found (check paths / CSV parsing).")

    print("=" * 60)
    print(f"COCO annotation file saved to → {output_json}\n")


# Example usage
if __name__ == "__main__":
    figlib_to_coco(
        csv_dir="/path/to/figlib/annotations",
        images_root="/path/to/figlib/images",
        output_json="/path/to/output/json",
    )
