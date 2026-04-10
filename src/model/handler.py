import torch
import torch.nn as nn
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import joblib
import os
from model import SimpleModel

class CustomHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")

        # Cargar modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel()
        model_path = os.path.join(model_dir, "simple_model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Cargar scalers
        self.scaler_x = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
        self.scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

    def preprocess(self, data):
        # Esperamos una lista de listas con valores numéricos
        input_data = data[0].get("body")
        if isinstance(input_data, (bytes, bytearray)):
            import json
            input_data = json.loads(input_data.decode("utf-8"))

        input_array = np.array(input_data)
        input_scaled = self.scaler_x.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)
        return input_tensor

    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.cpu().numpy()

    def postprocess(self, inference_output):
        # Desnormalizar salida
        output_inverse = self.scaler_y.inverse_transform(inference_output)
        return output_inverse.tolist()
