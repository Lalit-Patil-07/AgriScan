# Plant Disease Detection using CNN

Plant Disease Detection is an innovative machine learning project that harnesses the power of Convolutional Neural Networks (CNN) and deep learning techniques to identify and classify diseases in plants.

Try the app here: [Plant Disease Detection App](https://plant-disease-detection-7czedgzrnxqhvmvyhnwi8y.streamlit.app/)

## Dataset

The dataset used for this project can be found on Kaggle:
[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download).

The dataset is organized into subdirectories for each category of plant diseases and healthy plants. The data is split into training and validation datasets.

## Project Structure

The project comprises essential components:

- `Plant_disease(CNN).ipynb`: Jupyter Notebook with the code for model training.
- `main_app.py`: Streamlit web application for plant disease prediction.
- `plant_disease_model.h5`: Pre-trained model weights.
- `requirements.txt`: List of necessary Python packages.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

```bash
git clone <your-repository-link>
```

2. Navigate to the project directory:

```bash
cd <your-project-folder>
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit web application:

```bash
streamlit run main_app.py
```

## Usage

Once the application is running, open your web browser and navigate to [http://localhost:8501](http://localhost:8501). Upload an image of a plant leaf, and the system will predict if it is affected by any disease.

## Model Training

The model is a Sequential CNN model built using Keras/TensorFlow. It includes:

- Convolutional Layers
- Max Pooling Layers
- Dropout Layers
- Fully Connected Layers

The training process involves:

- Data Preprocessing
- Data Augmentation
- Model Compilation using Adam Optimizer
- Model Training and Evaluation

The model's performance is evaluated using metrics like accuracy and loss over training and validation datasets.

## Future Improvements

- Hyperparameter Tuning
- Using Transfer Learning
- Expanding the dataset with more classes

## License

This project is licensed under the MIT License.

