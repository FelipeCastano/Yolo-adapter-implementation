#!/bin/bash

# Moverse a la raíz del proyecto (asumiendo que el script está en model_store/)
cd "$(dirname "$0")/.."

echo "Creando archivo .mar desde la raíz del proyecto: $(pwd)"

# Crear carpeta model_store si no existe
mkdir -p model_store

# Ejecutar torch-model-archiver con rutas relativas a la raíz del proyecto
torch-model-archiver \
  --model-name simple_model \
  --version 1.0 \
  --serialized-file /app/model_store/simple_model.pt \
  --handler /app/model/handler.py \
  --model-file /app/model/model.py \
  --extra-files "/app/model_store/scaler_x.pkl,/app/model_store/scaler_y.pkl" \
  --export-path /app/model_store \
  --force


echo ".mar creado en model_store/"
