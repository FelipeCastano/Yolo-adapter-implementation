import os
import torch
import pandas as pd
import mlflow.pytorch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from model import SimpleModel
import joblib

# === Load dataset from CSV ===
df = pd.read_csv("/app/data/dataset.csv")
X = df.drop("MedHouseVal", axis=1).values  # target column
y = df["MedHouseVal"].values.reshape(-1, 1)

# === Preprocess ===
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Convert to tensors ===
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# === Model, loss, optimizer ===
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# === Training loop ===
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# === Evaluation ===
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()
    y_true = y_test.numpy()

    # Reverse scaling
    preds = scaler_y.inverse_transform(preds)
    y_true = scaler_y.inverse_transform(y_true)

    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    mse = loss.item()

    print(f"Final MSE Loss: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

# === Save model ===
os.makedirs("model_store", exist_ok=True)
#traced_model = torch.jit.trace(model, X_train)
#traced_model.save("model_store/simple_model.pt")
torch.save(model.state_dict(), "/app/model_store/simple_model.pt")
joblib.dump(scaler_x, "/app/model_store/scaler_x.pkl")
joblib.dump(scaler_y, "/app/model_store/scaler_y.pkl")

# === Log with MLflow ===
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="SimpleModel",
    input_example=X_test[:1].numpy()
)