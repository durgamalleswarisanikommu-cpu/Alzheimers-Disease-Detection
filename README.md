# Alzheimers-Disease-Detection
Automated Detection of Alzheimer's Disease Using Deep Learning on MRI Images with EfficientNetB0 and Streamlit.

# 🧠 Automated Detection of Alzheimer's Disease Using Deep Learning on MRI Images

An AI-powered application that uses Deep Learning and MRI brain images to classify Alzheimer's disease into four categories using the EfficientNetB0 model.

## 📌 Project Overview

Alzheimer's disease is a progressive neurological disorder that affects memory, thinking, and behavior.

This project aims to develop a deep learning-based system that analyzes MRI brain images and predicts the corresponding Alzheimer's disease category.

The system uses **Transfer Learning with EfficientNetB0** and provides an interactive interface built using Streamlit.

## 🎯 Objectives

- Detect Alzheimer's disease categories from MRI images.
- Apply deep learning techniques for medical image classification.
- Use transfer learning with the EfficientNetB0 architecture.
- Develop an interactive web application using Streamlit.
- Explore the application of AI in healthcare research.

## 🧠 Disease Classification

The model classifies MRI images into four categories:

1. NonDemented
2. VeryMildDemented
3. MildDemented
4. ModerateDemented

## 📂 Dataset

**Dataset:** Alzheimer's Multiclass Dataset (Equal and Augmented)

**Source:** Kaggle

https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented

The dataset contains MRI brain images organized into four classification categories.

## ⚙️ Methodology

1. **Data Collection:** Obtain MRI images from the Kaggle dataset.
2. **Preprocessing:** Convert images to RGB, resize them to 224 × 224 pixels, and prepare them for model input.
3. **Data Splitting:** Divide the dataset into training (80%), validation (10%), and testing (10%) sets.
4. **Model Development:** Use pretrained EfficientNetB0 with transfer learning.
5. **Classification:** Add fully connected layers and a Softmax output layer for four-class classification.
6. **Prediction:** Use the trained model to classify uploaded MRI images.

## 🏗️ Model Architecture

**EfficientNetB0 Transfer Learning**

- Input Image: 224 × 224 × 3
- Pretrained EfficientNetB0
- Global Average Pooling
- Batch Normalization
- Dropout (0.4)
- Dense Layer (256 neurons, ReLU)
- Output Layer (4 neurons, Softmax)

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model | EfficientNetB0 |
| Image Processing | Pillow |
| Data Analysis | NumPy |
| Web Framework | Streamlit |
| Development | Jupyter Notebook, Google Colab |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
Alzheimers-Disease-Detection/
│
├── app.py
├── Alzheimers_Disease_Detection.ipynb
├── alzheimer_model_oldd.keras
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/durgamalleswarisanikommu-cpu/Alzheimers-Disease-Detection.git
```

### 2. Navigate to the project directory

```bash
cd Alzheimers-Disease-Detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment (Windows)

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

## 📊 Model Evaluation

Model performance should be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

*Actual evaluation results will be added after verification of the model's test results.*

## 🔮 Future Enhancements

- Improve model performance through fine-tuning.
- Integrate Grad-CAM for visual model explainability.
- Explore additional deep learning architectures.
- Deploy the application on a cloud platform.
- Improve the user interface and prediction reports.

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a replacement for professional medical advice or clinical diagnosis.

## 👩‍💻 Author

**Sanikommu Durga Malleswari**

B.Tech Student | Computer Science & IT

GitHub: https://github.com/durgamalleswarisanikommu-cpu

---

⭐ If you find this project interesting, consider giving this repository a star!
