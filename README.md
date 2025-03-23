# 🚑 Fall Detection System

This project implements a **computer vision-based fall detection system** using deep learning techniques. The system can detect whether a person has fallen in images or video frames.

## 📁 Project Structure
- main.py - Main script for running fall detection and evaluation
- data_handler.py - Handles dataset loading and preprocessing
- my_models.py - Contains model definitions
- fall_dataset/ - Directory containing training and validation data
- images/ - Contains input images
- labels/ - Contains corresponding labels
- model_weights/ - Directory containing trained model weight files (.h5 files)


## 📊 Data Source
The dataset used in this project is sourced from **Kaggle**.

### 📄 Dataset Details
- **Name:** Fall Detection Dataset 
- **Link:** https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset
- **Size:** 374 samples  
- **Description:** *Images of people in various postures including standing, sitting, and fallen states in different environments*   

## 🧠 Models
The system uses two main models:

1. **Fall Detection Model (`fall_detection_model`)** - A custom deep learning model that classifies cropped person images as *fallen* or *not fallen*.  
2. **YOLO Model (`yolo_model`)** - Uses **YOLOv5** to detect people in images and generate bounding boxes.

## 📦 Dataset Labels
The `Fall_dataset` class processes the dataset with the following label convention:

- `0` - Fallen  
- `1` - Standing  
- `2` - Sitting  

## 🚀 Usage

### 🔍 Running Evaluation
This will:
- Load the fall detection model with pre-trained weights  
- Process validation images with YOLO to detect people  
- Apply the fall detection model to each detected person  
- Calculate performance metrics (accuracy, precision, recall, F1 score)  

### 🎓 Training a New Model
The system supports training using the `train` method in the `fall_detection_model` class.

## 📈 Performance Metrics
The system calculates:

- **Accuracy**: (TP + TN) / (TP + FP + TN + FN)  
- **Precision**: TP / (TP + FP)  
- **Recall**: TP / (TP + FN)  
- **F1 Score**: 2 * TP / (2 * TP + FP + FN)  

### Where:
- **TP** = True Positives (*Fall - Detected*)  
- **FP** = False Positives (*Not Fall - Detected*)  
- **TN** = True Negatives (*Not Fall - Not Detected*)  
- **FN** = False Negatives (*Fall - Not Detected*)  

## 🔧 Requirements
- **PyTorch**  
- **OpenCV**  
- **Matplotlib**  
- **NumPy**  
- **YOLOv5 (via torch hub)**  
- **PIL**  
