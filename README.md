

# **Skin Condition Classification using CNN**

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation & Dependencies](#installation--dependencies)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results & Performance](#results--performance)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

---

## **Project Overview**
This project focuses on **skin condition classification** using a **Convolutional Neural Network (CNN)**. It is designed to assist in **medical image analysis** by automatically identifying different skin diseases from images.

### **Key Features**
✅ Uses a **deep learning-based CNN model** for image classification.  
✅ Trained on a **diverse skin disease dataset**.  
✅ Implements **image augmentation** for better generalization.  
✅ Supports **multi-class classification** with multiple disease categories.  
✅ Achieves **high accuracy** using advanced deep learning techniques.  

---

## **Dataset**
- **Source:** [Skin Diseases Image Dataset - Kaggle](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)  
- **Categories:** The dataset consists of different skin conditions:  
  - **Eczema**
  - **Melanoma**
  - **Psoriasis**
  - **Vitiligo**
  - **Other conditions**
- **Dataset Preprocessing:**  
  - **Image resizing** to 224x224 pixels.  
  - **Normalization** for efficient training.  
  - **Data Augmentation** to reduce overfitting.  
  - **Splitting** into **80% training** and **20% testing** datasets.  

---

## **Installation & Dependencies**
To set up and run the project, install the required dependencies:  


pip install tensorflow keras numpy pandas matplotlib scikit-learn kaggle opencv-python flask


---

Project Structure

Skin_Condition_Classification/
│── dataset/                   # Extracted dataset folder  
│── models/                     # Saved trained models  
│── notebooks/                   # Jupyter notebooks for experimentation  
│── src/                         # Source code  
│   ├── data_preprocessing.py    # Handles dataset loading and preprocessing  
│   ├── model.py                 # Defines the CNN model architecture  
│   ├── train.py                 # Training script  
│   ├── evaluate.py              # Model evaluation and accuracy calculation  
│   ├── predict.py               # Inference script for new images  
│── app/                         # Deployment files  
│── README.md                    # Project documentation  
│── requirements.txt              # Required Python packages


---
Methodology

The project follows these steps:

1️⃣ Data Collection: Download and extract the dataset from Kaggle.
2️⃣ Data Preprocessing: Resize, normalize, and augment images.
3️⃣ Model Building: Design and compile a CNN architecture.
4️⃣ Training: Train the model using labeled images.
5️⃣ Evaluation: Assess model performance using test data.
6️⃣ Inference: Predict skin conditions on new images.
7️⃣ Deployment: Deploy the trained model as a web application.


---

Model Architecture

The CNN model follows this structure:


---

Training & Evaluation

The dataset is split into:

80% training data

20% testing data


Training parameters:

Loss function: Categorical Crossentropy

Optimizer: Adam

Batch size: 32

Epochs: 25



---

Results & Performance

The model achieves high accuracy on the test dataset:
✅ Training Accuracy: ~92%
✅ Validation Accuracy: ~88%
✅ Precision & Recall: High scores across all classes

Confusion Matrix Sample:


---
Future Improvements

✅ Use pretrained models (ResNet, EfficientNet) for better performance.
✅ Increase dataset size for more robust predictions.
✅ Implement real-time inference on mobile devices.
✅ Deploy as a cloud-based AI service.


---

Contributors

Amal M (Lead Developer & Researcher)

Dataset by Ismail ProMus (Kaggle)


---


License

This project is open-source under the MIT License. Feel free to modify and use it for research and development.

---

