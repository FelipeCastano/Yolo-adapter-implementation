import os
import cv2
import json
import shutil
import base64
import tempfile
import numpy as np
import albumentations as A
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split


class DataAugmenter:
    """Performs data augmentation on blood cell images (or similar datasets).

    Expects a pre-split dataset with the following structure::

        root/
          train/
            img/   ← JPEG images
            ann/   ← JSON annotations (e.g. BloodImage_00001.jpeg.json)
          val/
            img/
            ann/
          test/
            img/
            ann/

    JSON annotations follow the Supervisely-style rectangle format with
    absolute pixel coordinates. They are converted to YOLO format on the fly.

    Output is written to ``output_root/`` with the same train/val/test
    structure, using ``img/`` and ``ann/`` subfolders. Annotations are saved
    as YOLO ``.txt`` files.

    Only the **train** split is augmented. Val and test are converted and
    copied as-is.

    **Available operations in pipeline:**

    * **Orthogonal transforms:** ``RandomRotate90``, ``HorizontalFlip``,
      and ``VerticalFlip``.
    * **Morphological robustness:** High-sigma elastic transform.
    * **Noise & sharpness:** ``GaussNoise`` and ``Sharpen``.

    :param input_root: Root directory containing train/val/test subfolders.
    :type input_root: str
    :param output_root: Root output directory. Subfolders are created inside it.
    :type output_root: str
    :param class_map: Mapping from classTitle strings to integer YOLO class IDs.
        Example: ``{"RBC": 0, "WBC": 1, "Platelets": 2}``.
        If ``None``, class IDs are assigned alphabetically on first encounter.
    :type class_map: Dict[str, int] | None
    :param seed: Random seed for reproducibility.
    :type seed: int
    """

    def __init__(
        self,
        input_root: str,
        output_root: str,
        class_map: Dict[str, int] = None,
        seed: int = 42,
    ):
        self.input_root = input_root
        self.output_root = output_root
        self.seed = seed

        # Class mapping: built dynamically if not provided
        self.class_map: Dict[str, int] = class_map if class_map is not None else {}
        self._next_class_id = max(self.class_map.values(), default=-1) + 1

        self._setup_output_dirs()
        self._build_transform()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, iterations: int = 10):
        """Run the full pipeline.

        Augments the train split and converts + copies val/test.

        :param iterations: Number of augmented versions to generate per
            training image.
        :type iterations: int
        """
        for split in ("train", "val", "test"):
            pairs = self._get_pairs(split)
            if not pairs:
                print(f"[DataAugmenter] No pairs found for split '{split}', skipping.")
                continue

            if split == "train":
                print(f"[DataAugmenter] Augmenting {len(pairs)} train images × {iterations}")
                self._augment_split(pairs, split, iterations)
            else:
                print(f"[DataAugmenter] Copying {len(pairs)} {split} images (no augmentation)")
                self._convert_and_copy_split(pairs, split)

        # Save the final class map so downstream training knows the IDs
        self._save_class_map()
        print(f"[DataAugmenter] Done. Output: {self.output_root}")

    def generate_base64_variants(self, b64_image: str, n_variants: int = 5) -> List[str]:
        """Generate augmented versions of a Base64-encoded image (visual demo).

        :param b64_image: Source image encoded as a Base64 string.
        :type b64_image: str
        :param n_variants: Number of augmented variants to produce.
        :type n_variants: int
        :return: List of Base64-encoded augmented images.
        :rtype: List[str]
        """
        img_data = base64.b64decode(b64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = []
        for _ in range(n_variants):
            augmented = self.transform(image=image, bboxes=[], class_labels=[])
            _, buffer = cv2.imencode(".jpg", augmented["image"])
            results.append(base64.b64encode(buffer).decode("utf-8"))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_output_dirs(self):
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(self.output_root, split, sub), exist_ok=True)

    def _build_transform(self):
        self.transform = A.Compose(
            [
                # --- Geometric: orientation invariance ---
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # --- Illumination & color: simulate staining variability ---
                # Microscopy images vary in brightness and contrast depending
                # on the staining batch and microscope calibration.
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.5,
                ),
                # Hue/saturation shifts simulate different Giemsa stain
                # concentrations and slide preparation conditions.
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=0.4,
                ),

                # --- Blur: simulate focus variation across the slide ---
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=0.2,
                ),

                # --- Sharpness: simulate over-sharpened microscope output ---
                A.Sharpen(
                    alpha=(0.1, 0.25),
                    lightness=(0.8, 1.0),
                    p=0.2,
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3,
            ),
        )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _get_pairs(self, split: str) -> List[Dict]:
        """Return image/annotation pairs for a given split.

        :param split: One of ``'train'``, ``'val'``, ``'test'``.
        :type split: str
        :return: List of dicts with ``'image'`` and ``'annotation'`` paths.
        :rtype: List[Dict]
        """
        img_dir = os.path.join(self.input_root, split, "img")
        ann_dir = os.path.join(self.input_root, split, "ann")

        print(f"[DataAugmenter] img_dir: {img_dir} -> exists={os.path.isdir(img_dir)}")
        print(f"[DataAugmenter] ann_dir: {ann_dir} -> exists={os.path.isdir(ann_dir)}")

        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            return []

        all_files = os.listdir(img_dir)
        print(f"[DataAugmenter] Files in img_dir ({len(all_files)} total): {all_files[:5]}")

        pairs = []
        for fname in sorted(all_files):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, fname)
            # Annotation filename = image filename + ".json"
            ann_path = os.path.join(ann_dir, fname + ".json")
            if not os.path.exists(ann_path):
                print(f"[DataAugmenter] Warning: no annotation for {fname}, skipping.")
                continue
            pairs.append({"image": img_path, "annotation": ann_path})
        return pairs

    # ------------------------------------------------------------------
    # JSON → YOLO conversion
    # ------------------------------------------------------------------

    def _get_class_id(self, class_title: str) -> int:
        """Return (and register if new) the YOLO integer ID for a class title.

        :param class_title: Class name string from the JSON annotation.
        :type class_title: str
        :return: Integer class ID.
        :rtype: int
        """
        if class_title not in self.class_map:
            self.class_map[class_title] = self._next_class_id
            self._next_class_id += 1
        return self.class_map[class_title]

    def _json_to_yolo(
        self, ann_path: str, img_w: int, img_h: int
    ) -> Tuple[List[List[float]], List[int]]:
        """Parse a Supervisely-style JSON annotation and return YOLO bboxes.

        Coordinates in the JSON are absolute pixel values.
        YOLO format: ``[x_center, y_center, width, height]`` normalised to [0, 1].

        :param ann_path: Path to the ``.json`` annotation file.
        :type ann_path: str
        :param img_w: Image width in pixels.
        :type img_w: int
        :param img_h: Image height in pixels.
        :type img_h: int
        :return: Tuple of ``(bounding_boxes, class_labels)``.
        :rtype: Tuple[List[List[float]], List[int]]
        """
        with open(ann_path, "r") as f:
            data = json.load(f)

        bboxes, labels = [], []
        for obj in data.get("objects", []):
            if obj.get("geometryType") != "rectangle":
                continue  # skip non-rectangle geometries

            exterior = obj["points"]["exterior"]
            # exterior = [[x1, y1], [x2, y2]]
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]

            # Ensure correct ordering (just in case)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # Clamp to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)

            # Convert to YOLO normalised format
            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            width    = (x_max - x_min) / img_w
            height   = (y_max - y_min) / img_h

            if width <= 0 or height <= 0:
                continue

            class_id = self._get_class_id(obj["classTitle"])
            bboxes.append([x_center, y_center, width, height])
            labels.append(class_id)

        return bboxes, labels

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment_split(self, pairs: List[Dict], split: str, iterations: int):
        """Augment images and save results to the output split folder.

        :param pairs: List of dicts with ``'image'`` and ``'annotation'`` paths.
        :type pairs: List[Dict]
        :param split: Dataset split name (used to build the output path).
        :type split: str
        :param iterations: Number of augmented copies per image.
        :type iterations: int
        """
        out_img_dir = os.path.join(self.output_root, split, "images")
        out_ann_dir = os.path.join(self.output_root, split, "labels")

        for pair in pairs:
            image = cv2.imread(pair["image"])
            if image is None:
                print(f"[DataAugmenter] Could not read image: {pair['image']}, skipping.")
                continue

            img_h, img_w = image.shape[:2]
            bboxes, labels = self._json_to_yolo(pair["annotation"], img_w, img_h)

            base_name = os.path.splitext(os.path.basename(pair["image"]))[0]

            for i in range(iterations):
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=labels,
                )
                suffix = f"aug_{i:03d}"
                img_out = os.path.join(out_img_dir, f"{base_name}_{suffix}.jpg")
                ann_out = os.path.join(out_ann_dir, f"{base_name}_{suffix}.txt")

                cv2.imwrite(img_out, transformed["image"])
                self._write_yolo(ann_out, transformed["class_labels"], transformed["bboxes"])

    # ------------------------------------------------------------------
    # Copy without augmentation (val / test)
    # ------------------------------------------------------------------

    def _convert_and_copy_split(self, pairs: List[Dict], split: str):
        """Convert JSON annotations to YOLO and copy images unchanged.

        :param pairs: List of dicts with ``'image'`` and ``'annotation'`` paths.
        :type pairs: List[Dict]
        :param split: Dataset split name.
        :type split: str
        """
        out_img_dir = os.path.join(self.output_root, split, "images")
        out_ann_dir = os.path.join(self.output_root, split, "labels")

        for pair in pairs:
            image = cv2.imread(pair["image"])
            if image is None:
                print(f"[DataAugmenter] Could not read image: {pair['image']}, skipping.")
                continue

            img_h, img_w = image.shape[:2]
            bboxes, labels = self._json_to_yolo(pair["annotation"], img_w, img_h)

            fname = os.path.basename(pair["image"])
            base_name = os.path.splitext(fname)[0]

            shutil.copy(pair["image"], os.path.join(out_img_dir, fname))
            self._write_yolo(
                os.path.join(out_ann_dir, base_name + ".txt"),
                labels,
                bboxes,
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _write_yolo(self, path: str, labels: List[int], bboxes: List[List[float]]):
        """Write YOLO annotation file.

        :param path: Output ``.txt`` file path.
        :type path: str
        :param labels: List of integer class IDs.
        :type labels: List[int]
        :param bboxes: List of ``[x_center, y_center, width, height]`` in [0, 1].
        :type bboxes: List[List[float]]
        """
        with open(path, "w") as f:
            for label, bbox in zip(labels, bboxes):
                bbox_str = " ".join(f"{x:.6f}" for x in bbox)
                f.write(f"{label} {bbox_str}\n")

    def _save_class_map(self):
        """Save the class-ID mapping to ``classes.txt`` in the output root."""
        out_path = os.path.join(self.output_root, "classes.txt")
        # Sort by ID so the file is ordered 0, 1, 2, …
        sorted_classes = sorted(self.class_map.items(), key=lambda kv: kv[1])
        with open(out_path, "w") as f:
            for class_title, class_id in sorted_classes:
                f.write(f"{class_id}: {class_title}\n")
        print(f"[DataAugmenter] Class map saved to {out_path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    augmenter = DataAugmenter(
        input_root="/app/data/dataset",   # must contain train/, val/, test/
        output_root="/app/data/aug_data",
        # Optional: fix your class IDs explicitly.
        # If omitted, IDs are assigned alphabetically on first encounter.
        class_map={"RBC": 0, "WBC": 1, "Platelets": 2},
        seed=42,
    )
    augmenter.run(iterations=10)