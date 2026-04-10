import os
import cv2
import shutil
import numpy as np
import yaml


class PreprocessedDataset:
	"""Generates a preprocessed copy of a YOLO dataset with img/ann subdirectories."""

	def __init__(self, source_yaml: str, output_base: str = None):
		self._source_cfg = self._read_yaml(source_yaml)
		self._source_root = self._source_cfg["path"].rstrip("/")
		self._output_base = output_base or os.path.dirname(self._source_root)
		self._dataset_cfg = {
			"nc":    self._source_cfg["nc"],
			"names": self._source_cfg["names"],
		}

	def generate(self, output_name: str, op: str, binarize_threshold: int = 127, morph_kernel: int = 3, cfg_dir: str = "/app/model_storage/yolo_cfg") -> str:
		output_root = os.path.join(self._output_base, output_name)

		if os.path.exists(output_root):
			print(f"[PreprocessedDataset] Already exists, skipping: {output_root}")
			return os.path.join(cfg_dir, f"{output_name}.yaml")

		print(f"[PreprocessedDataset] Generating '{output_name}' with op='{op}'...")

		clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		
		# Define the splits based on the source YAML
		splits = {
			"train": self._source_cfg.get("train", "train"),
			"val":   self._source_cfg.get("val",   "val"),
			"test":  self._source_cfg.get("test",  "test"),
		}

		for split_subdir in splits.values():
			self._process_split(
				src_dir   = os.path.join(self._source_root, split_subdir),
				dst_dir   = os.path.join(output_root, split_subdir),
				op        = op,
				threshold = binarize_threshold,
				kernel    = morph_kernel,
				clahe_obj = clahe_obj,
			)

		self._write_yaml(output_root, splits)

		os.makedirs(cfg_dir, exist_ok=True)
		cfg_yaml_path = os.path.join(cfg_dir, f"{output_name}.yaml")
		shutil.copy(os.path.join(output_root, "dataset.yaml"), cfg_yaml_path)

		print(f"[PreprocessedDataset] Done → {cfg_yaml_path}")
		return cfg_yaml_path

	def _process_split(self, src_dir: str, dst_dir: str, op: str, threshold: int, kernel: int, clahe_obj) -> None:
		"""Process nested img and ann directories."""
		img_src = os.path.join(src_dir, "img")
		ann_src = os.path.join(src_dir, "ann")
		img_dst = os.path.join(dst_dir, "img")
		ann_dst = os.path.join(dst_dir, "ann")

		os.makedirs(img_dst, exist_ok=True)
		os.makedirs(ann_dst, exist_ok=True)

		# 1. Handle Annotations (Direct Copy)
		if os.path.exists(ann_src):
			for fname in os.listdir(ann_src):
				if fname.endswith('.txt'):
					shutil.copy(os.path.join(ann_src, fname), os.path.join(ann_dst, fname))

		# 2. Handle Images (Apply Operations)
		if os.path.exists(img_src):
			for fname in os.listdir(img_src):
				if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
					img = cv2.imread(os.path.join(img_src, fname))
					if img is None:
						continue
					
					processed_img = self._apply_op(img, op, threshold, kernel, clahe_obj)
					# Save as .jpg to maintain consistency
					out_name = os.path.splitext(fname)[0] + '.jpg'
					cv2.imwrite(os.path.join(img_dst, out_name), processed_img)

	def _apply_op(self, img: np.ndarray, op: str, threshold: int, kernel: int, clahe_obj) -> np.ndarray:
		if op == "binarize":
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			_, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
			return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
		if op == "morph_erode":
			k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
			return cv2.morphologyEx(img, cv2.MORPH_ERODE, k)
		if op == "morph_dilate":
			k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
			return cv2.morphologyEx(img, cv2.MORPH_DILATE, k)
		if op == "clahe":
			lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
			l, a, b = cv2.split(lab)
			lab = cv2.merge([clahe_obj.apply(l), a, b])
			return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
		return img

	def _write_yaml(self, output_root: str, splits: dict) -> str:
		"""Update YAML to point to the nested 'img' folders."""
		cfg = {
			"path":  output_root,
			"train": os.path.join(splits["train"], "img"),
			"val":   os.path.join(splits["val"], "img"),
			"test":  os.path.join(splits["test"], "img"),
			"nc":    self._dataset_cfg["nc"],
			"names": self._dataset_cfg["names"],
		}
		yaml_path = os.path.join(output_root, "dataset.yaml")
		with open(yaml_path, "w") as f:
			yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
		return yaml_path

	@staticmethod
	def _read_yaml(yaml_path: str) -> dict:
		with open(yaml_path, "r") as f:
			return yaml.safe_load(f)


if __name__ == "__main__":
	preprocessor = PreprocessedDataset(
		source_yaml = "/app/model_storage/yolo_cfg/dataset.yaml",
		output_base = "/app/data",
	)
	#preprocessor.generate(output_name="aug_data_binarize",     op="binarize",     binarize_threshold=127)
	preprocessor.generate(output_name="aug_data_morph_erode",  op="morph_erode",  morph_kernel=3)
	preprocessor.generate(output_name="aug_data_morph_dilate", op="morph_dilate", morph_kernel=3)
	preprocessor.generate(output_name="aug_data_clahe",        op="clahe")