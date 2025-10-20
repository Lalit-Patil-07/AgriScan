
import torch
import torch.nn as nn
from torchvision import models
import json
import sys
import os

print("--- [Export Script] Starting model export... ---")

# --- 1. Load Class Names to get num_classes ---
class_names_file = "class_names.json"
if not os.path.exists(class_names_file):
    print(f"Error: '{class_names_file}' not found.")
    print("Please place this script in the same directory as your 'class_names.json' file.")
    sys.exit(1)

try:
    with open(class_names_file, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"[Export Script] Model will be configured for {num_classes} classes.")
except Exception as e:
    print(f"[Export Script] Error reading 'class_names.json': {e}")
    sys.exit(1)


# --- 2. Define Model Architecture ---
model = models.resnet50() 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# --- 3. Load Your Trained Weights ---
model_weights_path = "plant_disease_resnet50_best.pth"
if not os.path.exists(model_weights_path):
    print(f"Error: Model weights '{model_weights_path}' not found.")
    print("Please place this script in the same directory as your weights file.")
    sys.exit(1)

print(f"[Export Script] Loading weights from {model_weights_path}...")
try:
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
except Exception as e:
    print(f"[Export Script] Error loading model weights: {e}")
    sys.exit(1)

# --- 4. Set to Evaluation Mode ---
model.eval()
print("[Export Script] Model set to evaluation mode.")

# --- 5. Trace the Model ---
IMG_SIZE = 256
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device='cpu')
print(f"[Export Script] Tracing model with dummy input shape: {dummy_input.shape}")
try:
    traced_model = torch.jit.trace(model, dummy_input)
except Exception as e:
    print(f"[Export Script] Error during model tracing: {e}")
    sys.exit(1)

# --- 6. Save the TorchScript Model ---
output_path = "plant_disease_model.pt"
traced_model.save(output_path)

print("-" * 50)
print(f"[Export Script] Successfully exported model to: {output_path}")
print("--- [Export Script] Finished ---")
