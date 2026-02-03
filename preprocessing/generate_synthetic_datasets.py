"""
NOTE:
This script contains local absolute paths used during thesis experiments.
To rerun the code, adapt the path definitions below
(e.g., json_path, images_root) to match your local directory structure.
"""

import json
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import albumentations as A
import numpy as np

#----------
# CONFIG
#----------

JSON_PATH = Path(
    "/path/to/pyro-sdis/splits/test.json"
)

IMAGES_ROOT = Path(
    "/path/to/pyro-sdis/images"
)

OUTPUT_ROOT = Path(
    "/path/to/pyro-sdis-synthetic-noise"
)

def apply_synthetic_shifts(image: np.ndarray, mode: str) -> np.ndarray:

    if mode == "fog":
        aug = A.Compose(
            [
                A.RandomFog(
                    fog_coef_range=(0.25, 0.35),
                    alpha_coef=0.05,
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=2,
                    sat_shift_limit=(-10, 5),
                    val_shift_limit=(-5, 3),
                    p=0.4,
                ),
            ]
        )
        return aug(image=image)["image"]

    elif mode == "lowlight":
        aug = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.15, -0.10),
                    contrast_limit=(-0.20, -0.15),
                    p=1.0,
                ),
                A.GaussNoise(
                    std_range=(0.03, 0.06),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    p=0.3,
                ),
            ]
        )
        return aug(image=image)["image"]

    elif mode == "noise":
        image_float = image.astype(np.float32) / 255.0
        aug = A.Compose(
            [
                A.GaussNoise(
                    std_range=(0.06, 0.10),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    p=1.0,
                )
            ]
        )
        augmented = aug(image=image_float)["image"]
        return np.clip(augmented * 255.0, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")


def generate_synthetic_dataset(
        json_path: Path, images_root: Path, output_root: Path, mode: str
) -> None:

    output_dir = output_root / mode
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data: Dict = json.load(f)

    processed_count = 0
    missing_count = 0

    for ann in tqdm(data["images"], desc=f"Processing {mode} augmentation"):
        img_name = Path(ann["file_name"]).name
        img_path = images_root / img_name

        if not img_path.exists():
            print(f"Missing image: {img_path}")
            missing_count += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read image: {img_path}")
            missing_count += 1
            continue

        image_augmented = apply_synthetic_shifts(image, mode)

        output_path = output_dir / "images" / img_name
        cv2.imwrite(str(output_path), image_augmented)

        ann["file_name"] = f"images/{img_name}"
        processed_count += 1

    annotation_path = output_dir / "annotations.json"
    with open(annotation_path, "w") as f:
        json.dump(data, f, indent=2)

    print(
        f"\nSynthetic dataset '{mode}' complete:\n"
        f"  Processed: {processed_count} images\n"
        f"  Missing:   {missing_count} images\n"
        f"  Output:    {output_dir}\n"
    )

if __name__ == "__main__":
    generate_synthetic_dataset(JSON_PATH, IMAGES_ROOT, OUTPUT_ROOT, mode="noise") # fog / lowlight / noise
