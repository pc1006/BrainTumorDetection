# **Brain Tumor Detection using Deep Learning**

This repository contains the implementation of a **Convolutional Neural Network (CNN) model** for brain tumor detection using MRI images. The model is designed to automate tumor detection with high accuracy, reducing the dependency on manual diagnosis by radiologists.

---

## **Introduction**

Brain tumors result from abnormal cell growth in the brain, which can severely impact an individual's health. Traditional methods for detecting brain tumors are time-consuming and prone to human errors. This project proposes a **deep learning-based CNN model** to detect brain tumors from MRI scans efficiently.

The model consists of **multiple convolutional layers** followed by pooling layers to extract essential features from MRI images. It has been trained and evaluated on a dataset containing **5,060 MRI images**, achieving an accuracy of **97%**.

---

## **Features**

- **Automated brain tumor detection** using deep learning.
- **High accuracy (97%)** in classification.
- **Segmentation of tumor region** using multi-level thresholding.
- **Real-time performance analysis** with precision, recall, and F1-score.
- **Keras-based CNN implementation** for easy deployment.

---

## **Dataset**

The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/datasets/jayaprakashpondy/brain-tumor-dataset). It contains **5,060 MRI images**, divided into an **80/20 training-validation split**, with an additional test set for predictions.

---

## **How to Run**

### **Prerequisites**

1. Install **Python 3.8+** on your system.
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Steps**

1. **Preprocess the MRI Images**:
   ```bash
   python preprocess.py
   ```
2. **Train the CNN Model**:
   ```bash
   python train.py
   ```
3. **Evaluate the Model**:
   ```bash
   python evaluate.py
   ```
4. **Make Predictions on New MRI Scans**:
   ```bash
   python predict.py --image <path_to_image>
   ```

---

## **Model Architecture**

The proposed CNN model consists of the following layers:
- **Convolutional Layers**: Extracts spatial features from MRI images.
- **MaxPooling Layers**: Reduces dimensionality while retaining essential features.
- **Flatten Layer**: Converts the feature maps into a single vector.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout Layers**: Reduces overfitting.

---

## **Experiments and Observations**

### **Baseline Model: Simple CNN**
- Accuracy: **85%**
- Observation: The model showed decent performance but lacked fine-grained feature extraction.

### **Proposed Model: CNN with Multi-Level Thresholding**
- Accuracy: **97%**
- Observation: The integration of segmentation and improved feature extraction enhanced performance significantly.

---

## **Performance Analysis**

The model's effectiveness is evaluated using the following metrics:

- **Accuracy**: 97%
- **Precision, Recall, F1-Score**: Provides insights into classification performance.
- **Confusion Matrix**: Analyzes misclassifications.

---

## **Technologies Used**

- **Python 3.8+**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib for Visualization**
- **Flask (for Deployment - Future Enhancement)**

---

## **Future Enhancements**

- Implementing real-time MRI scan analysis.
- Extending the model to other medical imaging datasets.
- Deploying as a **web-based application** using Flask.

---

## **Contributors**

- **G. Pavan Chaitanya**
- **Papishetty Prathima**
- **Pallala Priskilla**

Under the guidance of **Mrs. Ch Bhavani, Sr. Assistant Professor** at **CVR College of Engineering**.

---
