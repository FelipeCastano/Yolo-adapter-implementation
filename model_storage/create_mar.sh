#!/bin/bash
cd "$(dirname "$0")/.."
echo "Creating .mar from project root: $(pwd)"

mkdir -p model_storage

torch-model-archiver \
  --model-name  blood_cell_detection \
  --version     1.0 \
  --handler     /app/model/handler.py \
  --extra-files "/app/model/adapter_manager.py,\
/app/model/YOLO_adapter.py,\
/app/model/custom_trainer.py,\
/app/model_storage/yolo_cfg/yolov8s.pt,\
/app/model_storage/adapter_weights.pt" \
  --export-path /app/model_storage \
  --force
  
echo ".mar created in model_storage/"