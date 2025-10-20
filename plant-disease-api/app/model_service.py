import torch
import json
from torchvision import transforms
from PIL import Image
import io
import os
from .schemas import DiseaseInfo # Import our pydantic model

class ModelService:
    def __init__(self, model_path: str, class_names_path: str, knowledge_base_path: str):
        print("Initializing ModelService...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Load the optimized TorchScript model
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading TorchScript model: {e}")
            raise

        # 2. Load class names
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Successfully loaded {len(self.class_names)} class names.")
        except Exception as e:
            print(f"Error loading class names: {e}")
            raise

        # 3. Load knowledge base
        try:
            with open(knowledge_base_path, 'r') as f:
                self.knowledge_base = json.load(f)
            print("Successfully loaded knowledge base.")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.knowledge_base = {} # Continue without it

        # 4. Define the *exact* preprocessing pipeline from the notebook
        IMG_SIZE = 256
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            tensor = self.transform(image)
            return tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            raise ValueError("Could not process the uploaded image.")

    def _get_disease_info(self, disease_name: str) -> DiseaseInfo:
        """Fetches detailed disease info from the loaded knowledge base."""
        info_data = self.knowledge_base.get(disease_name, self.knowledge_base.get("Default", {}))
        
        return DiseaseInfo(
            description=info_data.get("description", "No description available."),
            symptoms=info_data.get("symptoms", ["No symptom information available."]),
            prevention=info_data.get("prevention", ["No prevention tips available."])
        )

    def predict(self, image_bytes: bytes) -> dict:
        input_tensor = self._preprocess(image_bytes)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_index = torch.max(probabilities, 1)
        
        disease_name = self.class_names[pred_index.item()]
        confidence_score = confidence.item()
        disease_details = self._get_disease_info(disease_name)

        return {
            "disease_name": disease_name,
            "confidence": confidence_score,
            "details": disease_details
        }

# --- Singleton Instance ---
print("Creating global ModelService instance...")
MODEL_PATH = os.getenv("MODEL_PATH", "plant_disease_model.pt")
CLASSES_PATH = os.getenv("CLASSES_PATH", "class_names.json")
KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
model_service = ModelService(
    model_path=MODEL_PATH,
    class_names_path=CLASSES_PATH,
    knowledge_base_path=KB_PATH
)