
# DeepSense HAR – Human Activity Recognition with Deep Learning


## Project Overview
This repository contains a **Human Activity Recognition (HAR)** web application powered by an advanced deep learning architecture that classifies physical activities from wearable device sensor data.  
The model processes accelerometer readings (x, y, z) and predicts the performed activity with high accuracy.  

The **front-end** is built using **Streamlit** for an intuitive, interactive experience, while **TensorFlow** powers the **back-end** for training and inference.

### The application can detect and classify the following activities:
- Walking  
- Jogging  
- Sitting  
- Standing  
- Going Upstairs  
- Going Downstairs  

---

## Technical Highlights
- **Upload and Predict**: Users can upload CSV files containing accelerometer sensor readings for classification.  
- **Real-Time Inference**: Provides activity predictions along with confidence scores for each class.  
- **Data Visualization**: Generates an interactive activity timeline to visualize detected activities over time.  
- **Downloadable Results**: Users can download prediction outputs for further analysis.  
- **Hybrid Deep Learning Model**: Combines **Convolutional Neural Networks (CNN)** for feature extraction, **Gated Recurrent Units (GRU)** for temporal sequence modeling, and **Attention Mechanisms** for focusing on relevant time steps.  
- **Dataset**: Trained on the **WISDM v1.1** (Wireless Sensor Data Mining) dataset, a benchmark dataset for wearable sensor-based activity recognition.  

---

## Potential Applications
- **Fitness Tracking**: Integration into health apps for step/activity monitoring.  
- **Elderly Care**: Automatic monitoring of activity levels to detect sedentary patterns or falls.  
- **Sports Analytics**: Tracking performance and movement patterns of athletes.  
- **Research & Development**: Useful for academic projects in IoT, wearable computing, and healthcare AI.  

---

## Key Benefits
- Combines deep learning accuracy with an easy-to-use web interface.  
- Offers both research-grade performance and real-world usability.  
- Modular design allows developers to retrain the model on different datasets or integrate additional activity classes.  

---

This project serves as a practical demonstration of **AI in action**, showing how **deep learning**, **time-series analysis**, and **web deployment** can work together to create an **end-to-end intelligent application**.


## How to use 
## 🛠 How to Use

1. Launch the app:
```bash
streamlit run app.py
```
1. Upload a CSV file containing accelerometer data with columns: X, Y, Z.

2. View predicted activities in real-time.
3. Download prediction results as a CSV file.

