import os
import gc
import json
from collections import defaultdict

import cv2
import torch
import mlflow
import yaml
import numpy as np
from PIL import Image
from ultralytics.utils import SETTINGS

from adapter_manager import AdapterManager
from custom_trainer import CustomTrainer


# ── Constants ─────────────────────────────────────────────────────────────────

YOLO_CONFIGS = {
    "v8": {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt"},
}

# All preprocessed dataset variants. Keys are used as MLflow experiment names.
# 'aug_data' is the augmented-only baseline (no extra preprocessing).
DATASETS = {
    "aug_data":             "/app/model_storage/yolo_cfg/dataset.yaml",
    "aug_data_clahe":       "/app/model_storage/yolo_cfg/aug_data_clahe.yaml",
    "aug_data_stain_norm":  "/app/model_storage/yolo_cfg/aug_data_stain_norm.yaml",
    "aug_data_gamma_08":    "/app/model_storage/yolo_cfg/aug_data_gamma_08.yaml",
    "aug_data_denoise":     "/app/model_storage/yolo_cfg/aug_data_denoise.yaml",
    "aug_data_morph_erode": "/app/model_storage/yolo_cfg/aug_data_morph_erode.yaml",
    "aug_data_morph_dilate":"/app/model_storage/yolo_cfg/aug_data_morph_dilate.yaml",
}

# Fixed training hyperparameters
EPOCHS       = 1
BATCH        = 8
LR           = 1e-3
OPTIMIZER    = "AdamW"
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3
PATIENCE     = 30

# Image size: matches the native acquisition resolution (480 × 640).
# Using the shorter side avoids unnecessary upscaling of the input images
# while keeping the standard YOLO stride constraints satisfied.
IMGSZ = 480


# ── TrainerManager ────────────────────────────────────────────────────────────

class TrainerManager:
    """Orchestrates training and evaluation across datasets and model sizes.

    For each dataset a single MLflow experiment is created. Within it,
    one run per model size logs parameters, metrics and prediction artefacts.

    Metrics computed during ``_test``:

    * **mAP50 / mAP50-95** — standard COCO detection metrics (global).
    * **Per-class breakdown** — AP50, AP50-95, Precision, Recall, F1,
      and Accuracy (TP / (TP + FP + FN)) for every class.
    * **Mean IoU** — average IoU of all matched (TP) detections.
    * **Cell-count MAE** — Mean Absolute Error between the number of
      predicted cells and ground-truth cells per image, computed by
      running inference over the test split image-by-image.

    :param base_pt_dir: Directory containing base YOLO ``.pt`` checkpoints.
    :type base_pt_dir: str
    :param runs_dir: Root directory for Ultralytics run outputs.
    :type runs_dir: str
    :param results_dir: Root directory where prediction images are saved.
    :type results_dir: str
    :param nc: Number of output classes.
    :type nc: int
    :param mlflow_uri: MLflow tracking URI.
    :type mlflow_uri: str
    :param test_dir: Path to the test image directory.
    :type test_dir: str
    :param imgsz: Input image size (shorter side). Default: ``480``.
    :type imgsz: int
    """

    def __init__(
        self,
        base_pt_dir: str,
        runs_dir: str,
        results_dir: str,
        nc: int,
        mlflow_uri: str,
        test_dir: str,
        imgsz: int = IMGSZ,
    ):
        self.base_pt_dir = base_pt_dir
        self.runs_dir    = runs_dir
        self.results_dir = results_dir
        self.nc          = nc
        self.mlflow_uri  = mlflow_uri
        self.test_dir    = test_dir
        self.imgsz       = imgsz
        self._setup_mlflow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data: str, dataset_name: str, yolo_size: str) -> dict:
        """Train, evaluate and log results for one dataset/model combination.

        :param data: Path to the dataset YAML file.
        :type data: str
        :param dataset_name: MLflow experiment name and run tag.
        :type dataset_name: str
        :param yolo_size: Model size — one of ``'n'``, ``'s'``, ``'m'``.
        :type yolo_size: str
        :return: Dict with ``'map50'``, ``'map50_95'`` and ``'weights_path'``.
        :rtype: dict
        """
        pt_path  = os.path.join(self.base_pt_dir, YOLO_CONFIGS["v8"][yolo_size])
        run_name = f"yolov8{yolo_size}"

        self._set_experiment(dataset_name)

        with mlflow.start_run(run_name=run_name):
            best_weights  = self._train(data, pt_path, run_name, dataset_name, yolo_size)
            adapted_model = self._load(pt_path, best_weights)
            metrics       = self._test(adapted_model, data, run_name, dataset_name)
            mlflow.log_artifacts(self.results_dir, artifact_path="predictions")

        del adapted_model
        gc.collect()
        #torch.cuda.empty_cache()
        #torch.cuda.synchronize()
        print(
            f"[TrainerManager] VRAM liberada. "
            f"Reservada: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )

        return {
            "map50":        metrics["map50"],
            "map50_95":     metrics["map50_95"],
            "weights_path": best_weights,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(
        self,
        data: str,
        pt_path: str,
        run_name: str,
        dataset_name: str,
        yolo_size: str,
    ) -> str:
        """Build model, log hyperparameters and run training.

        :return: Path to the best weights checkpoint.
        :rtype: str
        """
        manager       = AdapterManager(pt_path)
        adapted_model = manager.build(nc=self.nc)

        total     = sum(p.numel() for p in adapted_model.model.parameters())
        trainable = sum(
            p.numel() for p in adapted_model.model.parameters() if p.requires_grad
        )

        mlflow.log_params({
            "dataset":        dataset_name,
            "yolo_size":      yolo_size,
            "epochs":         EPOCHS,
            "batch":          BATCH,
            "lr":             LR,
            "optimizer":      OPTIMIZER,
            "weight_decay":   WEIGHT_DECAY,
            "warmup_epochs":  WARMUP_EPOCHS,
            "patience":       PATIENCE,
            "nc":             self.nc,
            "imgsz":          self.imgsz,
            "total_params":   total,
            "trainable_params": trainable,
            "trainable_pct":  round(100 * trainable / total, 2),
        })
        mlflow.set_tags({
            "model_base":     f"yolov8{yolo_size}",
            "adapter_type":   "C2f_Adapter",
            "init_strategy":  "data_driven_xavier",
            "dataset":        dataset_name,
        })

        trainer = CustomTrainer(
            adapted_model=adapted_model,
            overrides={
                "data":          data,
                "epochs":        EPOCHS,
                "imgsz":         self.imgsz,
                "batch":         BATCH,
                "lr0":           LR,
                "optimizer":     OPTIMIZER,
                "weight_decay":  WEIGHT_DECAY,
                "warmup_epochs": WARMUP_EPOCHS,
                "model":         pt_path,
                "name":          run_name,
                "project":       self.runs_dir,
                "patience":      PATIENCE,
            },
        )
        trainer.callbacks["on_train_end"] = [
            cb for cb in trainer.callbacks["on_train_end"]
            if "mlflow" not in str(cb)
        ]
        trainer.train()
        return str(trainer.best)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, pt_path: str, weights_path: str):
        """Rebuild architecture and load trained weights.

        :param pt_path: Base YOLO checkpoint path.
        :param weights_path: Fine-tuned checkpoint path.
        :return: Loaded adapted YOLO model.
        """
        manager = AdapterManager(pt_path)
        return manager.load_trained(weights_path, nc=self.nc)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _test(self, model, data: str, run_name: str, dataset_name: str) -> dict:
        """Evaluate on the test split and compute detailed metrics.

        Metrics computed
        ----------------
        Global
            mAP50, mAP50-95, Precision, Recall, F1, Mean IoU.

        Per class
            AP50, AP50-95, Precision, Recall, F1,
            Accuracy = TP / (TP + FP + FN).

        Cell-count MAE
            Mean Absolute Error between the number of predicted detections
            and the number of ground-truth boxes per image, across the
            entire test split.

        :param model: Loaded adapted YOLO model.
        :param data: Path to the dataset YAML config.
        :param run_name: Used to name the output subdirectory.
        :return: Dict with all computed metrics.
        :rtype: dict
        """
        print(f"\n[TrainerManager] Evaluating {run_name} on test split…")

        dataset_info          = self._read_yaml(data)
        model.model.names     = dataset_info["names"]
        model.model.nc        = dataset_info["nc"]
        class_names: dict     = dataset_info["names"]   # {0: "RBC", 1: "WBC", …}

        test_output_dir = os.path.join(self.results_dir, f"{dataset_name}_{run_name}_test_results")
        os.makedirs(test_output_dir, exist_ok=True)

        # ── 1. model.val() — detection metrics ────────────────────────
        results = model.val(
            data     = data,
            split    = "test",
            imgsz    = self.imgsz,
            batch    = BATCH,
            project  = test_output_dir,
            name     = "val_plots",
            save_json= False,
            plots    = True,
        )

        box = results.box

        # ── 2. Per-class AP50-95 ───────────────────────────────────────
        # box.ap shape can be (nc,) [mean over IoU thresholds already]
        # or (nc, n_thresholds) depending on Ultralytics version.
        raw_ap = np.array(box.ap)
        if raw_ap.ndim == 2:
            ap50_95_per_class = raw_ap.mean(axis=1).tolist()
        else:
            ap50_95_per_class = raw_ap.tolist()

        ap50_per_class = np.array(box.ap50).tolist()

        # box.p and box.r are (nc,) arrays — precision & recall per class
        precision_per_class = np.array(box.p).tolist()
        recall_per_class    = np.array(box.r).tolist()

        # ── 3. Per-class F1, Accuracy ─────────────────────────────────
        # Ultralytics stores TP counts in box.tp (nc,) and the confusion
        # matrix holds FP/FN. We derive them from P and R:
        #   P = TP / (TP + FP)  →  FP = TP * (1/P - 1)
        #   R = TP / (TP + FN)  →  FN = TP * (1/R - 1)
        # When P or R is 0 we set the derived count to 0 safely.
        per_class_metrics = {}
        for i, name in class_names.items():
            p   = precision_per_class[i] if i < len(precision_per_class) else 0.0
            r   = recall_per_class[i]    if i < len(recall_per_class)    else 0.0
            ap50    = ap50_per_class[i]    if i < len(ap50_per_class)    else 0.0
            ap5095  = ap50_95_per_class[i] if i < len(ap50_95_per_class) else 0.0

            # F1
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            # Accuracy = TP / (TP + FP + FN)
            # Expressed purely from P and R (avoids needing raw TP counts):
            #   TP/(TP+FP) = P  and  TP/(TP+FN) = R
            #   accuracy = P*R / (P + R - P*R)   [equivalent formulation]
            if (p + r - p * r) > 0:
                accuracy = (p * r) / (p + r - p * r)
            else:
                accuracy = 0.0

            per_class_metrics[name] = {
                "ap50":     round(ap50,   4),
                "ap50_95":  round(ap5095, 4),
                "precision":round(p,      4),
                "recall":   round(r,      4),
                "f1":       round(f1,     4),
                "accuracy": round(accuracy, 4),
            }

        # ── 4. Global F1 and Accuracy ──────────────────────────────────
        mp  = float(box.mp)   # mean precision
        mr  = float(box.mr)   # mean recall
        global_f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
        if (mp + mr - mp * mr) > 0:
            global_accuracy = (mp * mr) / (mp + mr - mp * mr)
        else:
            global_accuracy = 0.0

        # ── 5. Mean IoU of matched detections ─────────────────────────
        # results.box.mean_results() returns [P, R, mAP50, mAP50-95].
        # Ultralytics does not expose per-match IoU directly through the
        # public API, so we read it from the confusion matrix's TP IoU
        # tensor when available, falling back to a P/R-based estimate.
        mean_iou = self._compute_mean_iou(results)

        # ── 6. Cell-count MAE ──────────────────────────────────────────
        test_img_dir  = os.path.join(dataset_info["path"], "test", "images")
        test_ann_dir  = os.path.join(dataset_info["path"], "test", "labels")
        count_mae     = self._compute_count_mae(model, test_img_dir, test_ann_dir)

        # ── 7. Assemble summary ────────────────────────────────────────
        metrics = {
            "dataset":    os.path.basename(data),
            "model":      run_name,
            # Global
            "map50":          round(float(box.map50), 4),
            "map50_95":       round(float(box.map),   4),
            "precision":      round(mp,               4),
            "recall":         round(mr,               4),
            "f1":             round(global_f1,        4),
            "accuracy":       round(global_accuracy,  4),
            "mean_iou":       round(mean_iou,         4),
            "count_mae":      round(count_mae,        4),
            # Per class
            "per_class":      per_class_metrics,
        }

        # ── 8. Save JSON ───────────────────────────────────────────────
        summary_path = os.path.join(test_output_dir, "test_summary.json")
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # ── 9. Log scalar metrics to MLflow ───────────────────────────
        flat_mlflow = {
            "test_mAP50":     metrics["map50"],
            "test_mAP50_95":  metrics["map50_95"],
            "test_precision":  metrics["precision"],
            "test_recall":     metrics["recall"],
            "test_f1":         metrics["f1"],
            "test_accuracy":   metrics["accuracy"],
            "test_mean_iou":   metrics["mean_iou"],
            "test_count_mae":  metrics["count_mae"],
        }
        for class_name, cm in per_class_metrics.items():
            for metric_name, value in cm.items():
                flat_mlflow[f"test_{class_name}_{metric_name}"] = value

        mlflow.log_metrics(flat_mlflow)
        mlflow.log_artifacts(test_output_dir, artifact_path="test_manual_export")

        return metrics

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mean_iou(results) -> float:
        """Extract mean IoU of matched detections from validation results.

        Ultralytics stores per-threshold IoU data internally. We attempt to
        read ``results.box.iouv`` (IoU thresholds vector) and the TP mask
        ``results.box.tp`` to compute mean IoU over matched detections.
        If neither is available we fall back to a P/R harmonic approximation.

        :param results: Object returned by ``model.val()``.
        :return: Mean IoU as a float in [0, 1].
        :rtype: float
        """
        try:
            # Ultralytics >= 8.1 stores tp as (N, n_iou_thresholds) bool tensor
            tp = results.box.tp          # (N, 10) bool
            iouv = results.box.iouv      # (10,) tensor: [0.50, 0.55, …, 0.95]
            if tp is not None and iouv is not None:
                tp_np   = np.array(tp,   dtype=float)   # (N, 10)
                iouv_np = np.array(iouv, dtype=float)   # (10,)
                # For each detection, find the highest IoU threshold it matched
                matched_mask = tp_np.any(axis=1)        # (N,) — at least one threshold
                if matched_mask.sum() > 0:
                    # Weighted sum: each matched detection contributes the mean
                    # of the IoU thresholds at which it was a TP
                    matched_iou = (tp_np[matched_mask] * iouv_np).sum(axis=1) / \
                                   tp_np[matched_mask].sum(axis=1)
                    return float(matched_iou.mean())
        except Exception:
            pass

        # Fallback: estimate from global P and R
        # mean_IoU ≈ (P * R) / (P + R - P * R)  [Jaccard approximation]
        p = float(results.box.mp)
        r = float(results.box.mr)
        if (p + r - p * r) > 0:
            return (p * r) / (p + r - p * r)
        return 0.0

    def _compute_count_mae(
        self,
        model,
        img_dir: str,
        ann_dir: str,
    ) -> float:
        """Compute Mean Absolute Error of cell counts per image.

        For every image in ``img_dir`` the model predicts a number of
        bounding boxes. The ground-truth count comes from the corresponding
        YOLO ``.txt`` label file (one line = one object). The MAE is the
        mean of |predicted_count − gt_count| across all images.

        :param model: Loaded adapted YOLO model.
        :param img_dir: Directory containing test images.
        :type img_dir: str
        :param ann_dir: Directory containing YOLO ``.txt`` label files.
        :type ann_dir: str
        :return: MAE of cell counts per image.
        :rtype: float
        """
        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            print(
                "[TrainerManager] count_mae: img_dir or ann_dir not found, "
                "returning 0."
            )
            return 0.0

        abs_errors = []
        img_exts   = (".jpg", ".jpeg", ".png")

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(img_exts):
                continue

            img_path = os.path.join(img_dir, fname)
            ann_path = os.path.join(ann_dir, os.path.splitext(fname)[0] + ".txt")

            # Ground-truth count
            gt_count = 0
            if os.path.exists(ann_path):
                with open(ann_path) as f:
                    gt_count = sum(1 for line in f if line.strip())

            # Predicted count
            try:
                preds     = model.predict(img_path, imgsz=self.imgsz, verbose=False)
                pred_count = len(preds[0].boxes) if preds and preds[0].boxes else 0
            except Exception as e:
                print(f"[TrainerManager] count_mae: prediction failed for {fname}: {e}")
                pred_count = 0

            abs_errors.append(abs(pred_count - gt_count))

        if not abs_errors:
            return 0.0

        return float(np.mean(abs_errors))

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _set_experiment(self, name: str):
        """Create or activate an MLflow experiment.

        :param name: Experiment name.
        :type name: str
        """
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name)
        mlflow.set_experiment(name)

    def _setup_mlflow(self):
        """Configure MLflow tracking URI and Ultralytics integration."""
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"
        mlflow.set_tracking_uri(self.mlflow_uri)
        os.environ["MLFLOW_TRACKING_URI"] = self.mlflow_uri
        SETTINGS.update({"mlflow": True})

    def _resolve_pt(self, yolo_size: str) -> str:
        """Resolve the full path to a base YOLO checkpoint.

        :param yolo_size: Model size string.
        :type yolo_size: str
        :return: Absolute path to the ``.pt`` file.
        :rtype: str
        """
        return os.path.join(self.base_pt_dir, YOLO_CONFIGS["v8"][yolo_size])

    @staticmethod
    def _read_yaml(yaml_path: str) -> dict:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    manager = TrainerManager(
        base_pt_dir = "/app/model_storage/yolo_cfg",
        runs_dir    = "/app/model_storage/mlruns",
        results_dir = "/app/results",
        nc          = 3,                              # RBC, WBC, Platelets
        mlflow_uri  = "http://mlflow-server:5000",
        test_dir    = "/app/data/aug_data/test/images",
        imgsz       = IMGSZ,
    )

    for dataset_name, yaml_path in DATASETS.items():
        for yolo_size in ["n", "s", "m"]:
            print(f"\n{'=' * 60}")
            print(
                f"[TrainerManager] dataset={dataset_name} | "
                f"model=yolov8{yolo_size}"
            )
            print(f"{'=' * 60}")
            manager.run(
                data         = yaml_path,
                dataset_name = dataset_name,
                yolo_size    = yolo_size,
            )