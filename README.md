# Plant Disease Detection using CNN

This project is a high-accuracy plant disease detection system deployed as a full-stack web application. It uses a **PyTorch-based Convolutional Neural Network (CNN)**, built on a **ResNet50** architecture, to classify 38 different plant diseases from leaf images. The application is served through a **Python (FastAPI) backend API** and consumed by a modern **React frontend**.

## Model Details

The model was trained using transfer learning on a pre-trained **ResNet50** architecture.

  * **Framework:** PyTorch
  * **Base Architecture:** ResNet50 (from `torchvision.models`)
  * **Performance:** Achieved **97.7% validation accuracy**
  * **Key Techniques:**
      * Transfer learning (only the final fully-connected layer was trained).
      * Data Augmentation (RandomFlip, Rotation, ColorJitter) to improve robustness.
      * Class Balancing using `WeightedRandomSampler` to handle imbalanced data.
  * **Trained Model:** The best model weights are saved in `plant_disease_resnet50_best.pth`.

## Dataset

The model was trained on the **New Plant Diseases Dataset (Augmented)**, which is publicly available on Kaggle. This dataset contains over 87,000 RGB images of plant leaves, categorized into 38 distinct classes (including healthy and diseased leaves).

  * **Source:** [Kaggle: New Plant Diseases Dataset (Augmented)](https://www.google.com/search?q=https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset-augmented)
  * **Classes:** 38

## Project Structure

The project is now divided into three main components:

  * `/model_training.ipynb`: The Jupyter notebook containing all steps for data loading, preprocessing, model training, and evaluation.
  * `/plant-disease-api/`: The backend Python API (FastAPI) that serves the trained PyTorch model.
  * `/plant-disease-ui/`: The new frontend web application built with React.

## How to Run the New Application

To run the complete application, you must start both the backend API and the frontend UI.

### 1\. Backend (API)

(Instructions based on the `plant-disease-api` folder)

```bash
# Navigate to the API directory
cd plant-disease-api

# Install required Python packages
pip install -r requirements.txt

# Run the API server (using uvicorn)
uvicorn app.main:app --reload
```

### 2\. Frontend (React App)

(Instructions from the `plant-disease-ui/README.md`)

In a **new terminal**:

```bash
# Navigate to the UI directory
cd plant-disease-ui

# Install node modules
npm install

# Run the app in development mode
npm start
```

Once both services are running, you can view the app in your browser at [http://localhost:3000](https://www.google.com/search?q=http://localhost:3000).