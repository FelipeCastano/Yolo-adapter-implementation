import os
import cv2
import shutil
import numpy as np
import yaml


class PreprocessedDataset:
    """Generates a preprocessed copy of a YOLO dataset for blood-cell microscopy.

    Expects the source dataset to follow standard YOLO layout::

        source_root/
          train/
            images/
            labels/
          val/
            images/
            labels/
          test/
            images/
            labels/

    Each call to :meth:`generate` produces a new dataset variant under
    ``output_base/<output_name>/`` and writes the corresponding YOLO
    ``<output_name>.yaml`` config to ``cfg_dir/``.

    **Available preprocessing operations:**

    * ``clahe``          — Contrast-Limited Adaptive Histogram Equalisation in
                           LAB space. Improves local contrast without saturating
                           colours. Recommended first choice for Giemsa-stained slides.
    * ``stain_normalize``— Normalises stain intensity to a fixed reference mean/std
                           in LAB space (simplified Macenko-style). Reduces batch-to-
                           batch colour variation caused by different staining protocols.
    * ``gamma``          — Power-law intensity correction. Values < 1 brighten dark
                           slides; values > 1 darken over-exposed ones.
    * ``denoise``        — Non-local means colour denoising. Removes sensor noise
                           while preserving cell membrane edges.
    * ``morph_erode``    — Morphological erosion with an elliptical kernel.
                           Slightly shrinks bright foreground regions. Useful to
                           separate touching cells in dense fields.
    * ``morph_dilate``   — Morphological dilation with an elliptical kernel.
                           Slightly expands bright foreground regions.
    * ``none``           — No-op: copies images unchanged (useful as baseline).

    .. note::
        ``binarize`` has been intentionally removed: it destroys all colour and
        tonal information that the detector relies on for cell classification.

    :param source_yaml: Path to the source YOLO dataset ``.yaml`` config file.
    :type source_yaml: str
    :param output_base: Parent directory for all generated variants. Defaults to
        the directory that contains the source dataset root.
    :type output_base: str | None
    """

    # Reference LAB statistics used by stain_normalize.
    # Derived from a representative Giemsa-stained blood smear.
    _STAIN_REF_MEAN = np.array([198.0, 135.0, 120.0], dtype=np.float32)
    _STAIN_REF_STD  = np.array([  25.0,  12.0,  14.0], dtype=np.float32)

    def __init__(self, source_yaml: str, output_base: str = None):
        self._source_cfg  = self._read_yaml(source_yaml)
        self._source_root = self._source_cfg["path"].rstrip("/")
        self._output_base = output_base or os.path.dirname(self._source_root)
        self._dataset_cfg = {
            "nc":    self._source_cfg["nc"],
            "names": self._source_cfg["names"],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        output_name: str,
        op: str,
        # CLAHE
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid:  tuple  = (8, 8),
        # Morphological ops
        morph_kernel: int = 3,
        # Gamma correction
        gamma: float = 1.0,
        # Denoising
        denoise_h:         int = 6,
        denoise_template:  int = 7,
        denoise_search:    int = 21,
        # Stain normalisation
        stain_ref_mean: np.ndarray = None,
        stain_ref_std:  np.ndarray = None,
        # Output location
        cfg_dir: str = "/app/model_storage/yolo_cfg",
    ) -> str:
        """Generate a preprocessed dataset variant.

        All splits (train / val / test) are processed identically so that
        inference conditions match training conditions.

        :param output_name: Name of the output subdirectory and YAML file.
        :type output_name: str
        :param op: Preprocessing operation to apply. One of ``'clahe'``,
            ``'stain_normalize'``, ``'gamma'``, ``'denoise'``,
            ``'morph_erode'``, ``'morph_dilate'``, ``'none'``.
        :type op: str
        :param clahe_clip_limit: CLAHE clip limit. Higher values increase
            contrast enhancement but risk noise amplification. Default ``3.0``.
        :type clahe_clip_limit: float
        :param clahe_tile_grid: CLAHE tile grid size ``(rows, cols)``.
            Default ``(8, 8)``.
        :type clahe_tile_grid: tuple
        :param morph_kernel: Side length of the elliptical structuring element
            for erosion/dilation. Default ``3``.
        :type morph_kernel: int
        :param gamma: Gamma value for power-law correction. ``< 1`` brightens,
            ``> 1`` darkens. Default ``1.0`` (identity).
        :type gamma: float
        :param denoise_h: Filter strength for non-local means denoising.
            Higher = smoother, but may blur fine cell detail. Default ``6``.
        :type denoise_h: int
        :param denoise_template: Template window size for denoising. Default ``7``.
        :type denoise_template: int
        :param denoise_search: Search window size for denoising. Default ``21``.
        :type denoise_search: int
        :param stain_ref_mean: Target LAB mean ``[L, A, B]`` for stain
            normalisation. Defaults to a built-in Giemsa reference.
        :type stain_ref_mean: np.ndarray | None
        :param stain_ref_std: Target LAB std ``[L, A, B]`` for stain
            normalisation. Defaults to a built-in Giemsa reference.
        :type stain_ref_std: np.ndarray | None
        :param cfg_dir: Directory where the output YAML config is written.
        :type cfg_dir: str
        :return: Absolute path to the generated YAML config file.
        :rtype: str
        """
        output_root = os.path.join(self._output_base, output_name)

        if os.path.exists(output_root):
            print(f"[PreprocessedDataset] Already exists, skipping: {output_root}")
            return os.path.join(cfg_dir, f"{output_name}.yaml")

        print(f"[PreprocessedDataset] Generating '{output_name}' with op='{op}'...")

        # Build per-op stateful objects once, reuse across all images
        clahe_obj = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )
        gamma_lut = self._build_gamma_lut(gamma)
        ref_mean  = stain_ref_mean if stain_ref_mean is not None else self._STAIN_REF_MEAN
        ref_std   = stain_ref_std  if stain_ref_std  is not None else self._STAIN_REF_STD

        for split in ("train", "val", "test"):
            self._process_split(
                split      = split,
                src_root   = self._source_root,
                dst_root   = output_root,
                op         = op,
                clahe_obj  = clahe_obj,
                gamma_lut  = gamma_lut,
                morph_kernel = morph_kernel,
                denoise_h        = denoise_h,
                denoise_template = denoise_template,
                denoise_search   = denoise_search,
                ref_mean   = ref_mean,
                ref_std    = ref_std,
            )

        os.makedirs(cfg_dir, exist_ok=True)
        cfg_yaml_path = os.path.join(cfg_dir, f"{output_name}.yaml")
        self._write_yaml(cfg_yaml_path, output_root)

        print(f"[PreprocessedDataset] Done → {cfg_yaml_path}")
        return cfg_yaml_path

    # ------------------------------------------------------------------
    # Split processing
    # ------------------------------------------------------------------

    def _process_split(
        self,
        split: str,
        src_root: str,
        dst_root: str,
        op: str,
        clahe_obj,
        gamma_lut: np.ndarray,
        morph_kernel: int,
        denoise_h: int,
        denoise_template: int,
        denoise_search: int,
        ref_mean: np.ndarray,
        ref_std: np.ndarray,
    ) -> None:
        img_src = os.path.join(src_root, split, "images")
        ann_src = os.path.join(src_root, split, "labels")
        img_dst = os.path.join(dst_root, split, "images")
        ann_dst = os.path.join(dst_root, split, "labels")

        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(ann_dst, exist_ok=True)

        # Labels: always copy as-is (preprocessing is image-only)
        if os.path.isdir(ann_src):
            for fname in os.listdir(ann_src):
                if fname.endswith(".txt"):
                    shutil.copy(
                        os.path.join(ann_src, fname),
                        os.path.join(ann_dst, fname),
                    )

        # Images: apply the selected operation
        if not os.path.isdir(img_src):
            print(f"[PreprocessedDataset] images dir not found, skipping: {img_src}")
            return

        for fname in os.listdir(img_src):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(img_src, fname))
            if img is None:
                print(f"[PreprocessedDataset] Could not read {fname}, skipping.")
                continue

            processed = self._apply_op(
                img,
                op            = op,
                clahe_obj     = clahe_obj,
                gamma_lut     = gamma_lut,
                morph_kernel  = morph_kernel,
                denoise_h         = denoise_h,
                denoise_template  = denoise_template,
                denoise_search    = denoise_search,
                ref_mean      = ref_mean,
                ref_std       = ref_std,
            )
            out_name = os.path.splitext(fname)[0] + ".jpg"
            cv2.imwrite(os.path.join(img_dst, out_name), processed)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _apply_op(
        self,
        img: np.ndarray,
        op: str,
        clahe_obj,
        gamma_lut: np.ndarray,
        morph_kernel: int,
        denoise_h: int,
        denoise_template: int,
        denoise_search: int,
        ref_mean: np.ndarray,
        ref_std: np.ndarray,
    ) -> np.ndarray:
        """Dispatch to the appropriate preprocessing method.

        :param img: Input BGR image as a NumPy array.
        :type img: np.ndarray
        :param op: Operation name string.
        :type op: str
        :return: Preprocessed BGR image.
        :rtype: np.ndarray
        """
        if op == "clahe":
            return self._clahe(img, clahe_obj)
        if op == "stain_normalize":
            return self._stain_normalize(img, ref_mean, ref_std)
        if op == "gamma":
            return self._gamma(img, gamma_lut)
        if op == "denoise":
            return self._denoise(img, denoise_h, denoise_template, denoise_search)
        if op == "morph_erode":
            return self._morph(img, cv2.MORPH_ERODE, morph_kernel)
        if op == "morph_dilate":
            return self._morph(img, cv2.MORPH_DILATE, morph_kernel)
        if op == "none":
            return img
        raise ValueError(
            f"[PreprocessedDataset] Unknown op '{op}'. "
            "Choose from: clahe, stain_normalize, gamma, denoise, "
            "morph_erode, morph_dilate, none."
        )

    @staticmethod
    def _clahe(img: np.ndarray, clahe_obj) -> np.ndarray:
        """Apply CLAHE to the L channel in LAB colour space.

        Operating in LAB isolates luminance from colour, so hue and
        saturation of the Giemsa stain are preserved while local contrast
        is improved.

        :param img: Input BGR image.
        :param clahe_obj: Pre-built ``cv2.CLAHE`` instance.
        :return: Contrast-enhanced BGR image.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge([clahe_obj.apply(l), a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _stain_normalize(
        img: np.ndarray,
        ref_mean: np.ndarray,
        ref_std: np.ndarray,
    ) -> np.ndarray:
        """Normalise stain intensity to a fixed LAB reference distribution.

        This is a simplified Macenko-style normalisation that works entirely
        in LAB space without SVD, making it fast and robust to outliers.
        It shifts and scales each LAB channel so that the image's mean and
        standard deviation match those of the reference slide.

        :param img: Input BGR image.
        :param ref_mean: Target LAB mean ``[L, A, B]``.
        :param ref_std: Target LAB std ``[L, A, B]``.
        :return: Stain-normalised BGR image.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        for ch in range(3):
            channel = lab[:, :, ch]
            ch_mean = channel.mean()
            ch_std  = channel.std() + 1e-6          # avoid division by zero
            # z-score then rescale to reference distribution
            lab[:, :, ch] = (channel - ch_mean) / ch_std * ref_std[ch] + ref_mean[ch]
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _gamma(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Apply power-law gamma correction via a precomputed LUT.

        :param img: Input BGR image.
        :param lut: 256-entry uint8 look-up table.
        :return: Gamma-corrected BGR image.
        """
        return cv2.LUT(img, lut)

    @staticmethod
    def _denoise(
        img: np.ndarray,
        h: int,
        template_window: int,
        search_window: int,
    ) -> np.ndarray:
        """Remove sensor noise with non-local means colour denoising.

        Preserves cell membrane edges better than simple Gaussian blur.

        :param img: Input BGR image.
        :param h: Filter strength (higher = smoother, may blur fine detail).
        :param template_window: Template patch size in pixels (odd number).
        :param search_window: Search region size in pixels (odd number).
        :return: Denoised BGR image.
        """
        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h,
            h,
            template_window,
            search_window,
        )

    @staticmethod
    def _morph(img: np.ndarray, morph_op: int, kernel_size: int) -> np.ndarray:
        """Apply morphological erosion or dilation with an elliptical kernel.

        An elliptical structuring element is more appropriate than a
        rectangular one for round/oval blood cells.

        :param img: Input BGR image.
        :param morph_op: OpenCV morphological operation constant
            (e.g. ``cv2.MORPH_ERODE`` or ``cv2.MORPH_DILATE``).
        :param kernel_size: Diameter of the elliptical kernel in pixels.
        :return: Morphologically processed BGR image.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        return cv2.morphologyEx(img, morph_op, kernel)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Build a 256-entry uint8 look-up table for gamma correction.

        :param gamma: Gamma exponent. ``< 1`` brightens, ``> 1`` darkens.
        :type gamma: float
        :return: 256-entry uint8 NumPy array.
        :rtype: np.ndarray
        """
        inv_gamma = 1.0 / max(gamma, 1e-6)
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return table

    def _write_yaml(self, yaml_save_path: str, dataset_root: str) -> None:
        """Write a YOLO-compatible dataset config YAML.

        :param yaml_save_path: Destination path for the ``.yaml`` file.
        :type yaml_save_path: str
        :param dataset_root: Absolute path to the processed dataset root.
        :type dataset_root: str
        """
        cfg = {
            "path":  dataset_root,
            "train": "train/images",
            "val":   "val/images",
            "test":  "test/images",
            "nc":    self._dataset_cfg["nc"],
            "names": self._dataset_cfg["names"],
        }
        with open(yaml_save_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def _read_yaml(yaml_path: str) -> dict:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    preprocessor = PreprocessedDataset(
        source_yaml  = "/app/model_storage/yolo_cfg/dataset.yaml",
        output_base  = "/app/data",
    )

    # CLAHE: best general-purpose choice for Giemsa-stained slides
    preprocessor.generate(
        output_name      = "aug_data_clahe",
        op               = "clahe",
        clahe_clip_limit = 3.0,
        clahe_tile_grid  = (8, 8),
    )

    # Stain normalisation: useful when slides come from multiple batches
    preprocessor.generate(
        output_name = "aug_data_stain_norm",
        op          = "stain_normalize",
        # Override reference stats here if you have a preferred reference slide:
        # stain_ref_mean = np.array([195.0, 133.0, 118.0]),
        # stain_ref_std  = np.array([22.0, 11.0, 13.0]),
    )

    # Gamma correction: brighten under-exposed slides
    preprocessor.generate(
        output_name = "aug_data_gamma_08",
        op          = "gamma",
        gamma       = 0.8,   # < 1 → brightens
    )

    # Denoising: reduce sensor noise before training
    preprocessor.generate(
        output_name      = "aug_data_denoise",
        op               = "denoise",
        denoise_h        = 6,
        denoise_template = 7,
        denoise_search   = 21,
    )

    # Morphological ops: use sparingly for dense cell separation
    preprocessor.generate(
        output_name  = "aug_data_morph_erode",
        op           = "morph_erode",
        morph_kernel = 3,
    )
    preprocessor.generate(
        output_name  = "aug_data_morph_dilate",
        op           = "morph_dilate",
        morph_kernel = 3,
    )