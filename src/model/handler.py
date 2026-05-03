import base64
import io
import json
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

sys.path.append(os.path.dirname(__file__))

from adapter_manager import AdapterManager


class YOLOAdapterHandler(BaseHandler):
    """TorchServe handler for the YOLO C2f-Adapter detection model.

    Applies binarization preprocessing before inference to match the
    training data distribution of the best performing model.

    Expected input::

        {"image": "<base64_encoded_image>"}

    Response::

        {
            "detections": [
                {
                    "class_id":   2,
                    "class_name": "RBC",
                    "confidence": 0.91,
                    "bbox": {"x1": 120, "y1": 45, "x2": 210, "y2": 98}
                }
            ]
        }
    """

    CLASS_NAMES        = ["RBC", "WBC", "Pl"]
    CONF_THRESHOLD     = 0.25
    IMGSZ              = 640
    BINARIZE_THRESHOLD = 127

    def initialize(self, context):
        """Load the adapted YOLO model from the MAR archive.

        :param context: TorchServe context object.
        """
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir    = context.system_properties.get("model_dir")
        pt_path      = os.path.join(model_dir, "yolov8s.pt")
        weights_path = os.path.join(model_dir, "adapter_weights.pt")

        print(f"[YOLOAdapterHandler] Loading model from {weights_path}")
        manager          = AdapterManager(pt_path)
        self.model       = manager.load_trained(weights_path, nc=len(self.CLASS_NAMES))
        self.model.model.to(self.device)
        self.model.model.eval()
        self.initialized = True
        print(f"[YOLOAdapterHandler] Ready on {self.device}")

    def preprocess(self, data) -> Image.Image:
        """Decode Base64 image and apply binarization.

        Converts to grayscale, applies fixed-threshold binarization to
        match the training preprocessing of the best performing model
        (yolov8m + binarize), then converts back to RGB for YOLO.

        :param data: List of request dicts with a ``body`` key.
        :return: Binarized PIL Image in RGB format.
        :rtype: PIL.Image.Image
        """
        body = data[0].get("body") or data[0].get("data")
        if isinstance(body, (bytes, bytearray)):
            body = json.loads(body.decode("utf-8"))

        b64 = body.get("image", "")
        if "," in b64:
            b64 = b64.split(",")[1]

        img           = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        img_np        = np.array(img)
        kernel_size = 3 
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img_dilated = cv2.dilate(img_np, kernel, iterations=1)
        return Image.fromarray(img_dilated)

    def inference(self, img: Image.Image):
        """Run YOLO detection on the preprocessed image.

        :param img: Binarized PIL Image in RGB format.
        :type img: PIL.Image.Image
        :return: Ultralytics prediction result object.
        """
        results = self.model.predict(
            source  = img,
            imgsz   = self.IMGSZ,
            conf    = self.CONF_THRESHOLD,
            device  = self.device,
            verbose = False,
        )
        return results[0]

    def postprocess(self, result) -> list:
        """Convert Ultralytics result to a JSON-serializable detection list.

        :param result: Single Ultralytics prediction result object.
        :return: List containing one JSON string with ``'detections'``.
        :rtype: list[str]
        """
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes   = result.boxes.xyxy.cpu().numpy()
            confs   = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append({
                    "class_id":   int(cls_id),
                    "class_name": self.CLASS_NAMES[cls_id],
                    "confidence": round(float(conf), 4),
                    "bbox": {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                    },
                })

        return [json.dumps({"detections": detections})]