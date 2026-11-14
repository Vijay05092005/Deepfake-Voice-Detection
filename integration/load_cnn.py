import os
import json
import torch
from integration.cnn_model import DeepfakeCNN


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "best_deepfake_cnn.pth"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "final_model_config.json"
)


# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load configuration
with open(CONFIG_PATH, "r") as file:
    config = json.load(file)


# Initialize CNN
model = DeepfakeCNN(num_classes=2)


# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()


# Decision threshold
THRESHOLD = config["optimized_threshold"]


print("=" * 50)
print("TRAINED CNN MODEL LOADED SUCCESSFULLY")
print("=" * 50)

print("Model:", config["model_name"])
print("Architecture:", config["architecture"])
print("Device:", DEVICE)
print("Best epoch:", checkpoint["epoch"])
print("Development Macro F1:", checkpoint["macro_f1"])
print("Decision threshold:", THRESHOLD)
print("Input shape: (1, 1, 128, 501)")

print("\nCNN MODEL IS READY FOR INFERENCE!")