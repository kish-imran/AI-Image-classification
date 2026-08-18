# 🤖 AI Image Classification System

## 📌 Project Description

# AI Image Classification System

AI Image Classification System is a Machine Learning project that uses a Convolutional Neural Network (CNN) to classify images into categories. It currently supports Cats vs Dogs classification. The system processes an input image and predicts its class with a confidence score.


---

## 🎯 Objectives

* Build an AI-based image classification system.
* Train a CNN model using image datasets.
* Classify images into Cats and Dogs.
* Preprocess images before prediction.
* Evaluate model accuracy.
* Save and reuse the trained model.
* Provide a simple command-line interface.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* CNN
* NumPy
* Pillow
* Computer Vision
* Machine Learning
* VS Code

---

## 📂 Project Structure

```text
AI-Image-Classification/
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   │
│   └── validation/
│       ├── cats/
│       └── dogs/
│
├── model/
│   └── image_classifier.keras
│
├── test_images/
│   └── test.jpg
│
├── train_model.py
├── predict.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Model

This project uses a **Convolutional Neural Network (CNN)**.

The CNN contains:

* Convolutional layers
* Max Pooling layers
* Flatten layer
* Dense layer
* Dropout layer
* Sigmoid output layer

The final output predicts whether the image belongs to:

```text
CAT
```

or

```
```
