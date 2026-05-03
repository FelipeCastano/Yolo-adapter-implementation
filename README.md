#  Blood Cell Detection

> Full pipeline for training, experimentation, and deployment of Blood Cell Detection model using **YOLO**, **Docker**, **MLflow**, and **TorchServe**.

---

## Pipeline Execution

### 1 · Data Generation

Run the data pipeline to generate augmented dataset, create splits, and preprocess datasets.

```bash
docker compose -f container/docker-compose.yml up data-pipeline
```

---

### 2 · Start MLflow

```bash
docker compose -f container/docker-compose.yml up -d mlflow-server
```

---

### 3 · Model Training

```bash
# CPU
docker compose -f container/docker-compose.yml up trainer trainer

# GPU
docker compose -f container/docker-compose.yml up trainer trainer-gpu
```

> **Notes**
> - After experimentation, manually select the best model.
> - If the selected model requires specific preprocessing, adjust it in the handler.

**Outputs**

| Artifact | Location |
|---|---|
| Plots | `results/` |
| MLflow experiments | `model_storage/mlruns/` |
| Model weights | `model_storage/mlruns/named_folders` *(e.g. `yolo8m_data_augmented`)* |

---

### Training Results
YOLOv8 Nano vs. Small vs. Medium

Evaluating all three model sizes (N, S, M) across the various augmentation strategies reveals clear patterns regarding which combinations yield the best performance.

#### Top Performers by Metric

1.  **Highest mAP50 (Standard Bounding Box Overlap):**
    * **1st:** YOLOv8s + Morph Dilate (**0.9203**)
    * **2nd:** YOLOv8m + Gamma 0.8 (0.9179)
    * **3rd:** YOLOv8s + Denoise (0.9178)

2.  **Highest F1 Score & Accuracy (Best Classification Balance):**
    * **1st:** YOLOv8s + Morph Erode (**F1: 0.8825, Acc: 0.7897**)
    * **2nd:** YOLOv8n + Gamma 0.8 (F1: 0.8809, Acc: 0.7871)
    * **3rd:** YOLOv8m + Denoise (F1: 0.8792, Acc: 0.7845)

3.  **Lowest Count MAE (Best for Object/Cell Counting):**
    * **1st:** YOLOv8m + CLAHE (**4.1389**)
    * **2nd:** YOLOv8m + Morph Erode (5.2500)
    * **3rd:** YOLOv8m + Gamma 0.8 (5.6250)

---

### Final Conclusion and Model Selection

The selection of the model requires aligning the specific needs of the project with the empirical data:

* **The Overall Best for General Detection:**
    **YOLOv8s trained on the Morph Dilate dataset.** This combination achieves the highest mAP50 (0.9203) across all tests, and the highest F1 Score.

### 4 · Model Build (MAR)

Copy the best weights into place, then build the model archive:

```bash
# Step 1 — copy weights
cp best.pt model_storage/adapter_weights.pt

# Step 2 — build MAR
chmod +x model_storage/create_mar.sh
docker compose -f container/docker-compose.yml up mar-builder
```

---

### 5 · Deploy with TorchServe

```bash
docker compose -f container/docker-compose.yml up -d torchserve
```

---

### 6 · TorchServe Testing

**List models**
```bash
curl http://localhost:8081/models
```

**Run inference**
```bash
curl -X POST http://localhost:8080/predictions/blood_cell_detection \
  -H "Content-Type: application/json" \
  -d @data/sample_input.json
```

---

### 7 · Launch Demo

```bash
docker compose -f container/docker-compose.yml up api frontend
```

| Service | URL |
|---|---|
| Backend | http://localhost:8000/docs |
| Frontend | http://localhost:8501/ |

---

## Frontend

The frontend includes three tabs:

| Tab | Description |
|---|---|
| **Demo** | Upload an image to detect and classify Blood Cell Detection |
| **Augmentation Example** | Shows how data augmentation works on uploaded images |
| **Documentation** | Sphinx-generated documentation explaining the model and augmentation process |

---

