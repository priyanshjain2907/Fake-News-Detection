# 📰 Fake News Detection System

A Machine Learning-based web application that classifies news content as **REAL** or **FAKE** using Natural Language Processing (NLP) techniques.

---

## 📌 Overview
This project focuses on detecting fake news by analyzing textual patterns in news articles and statements.  
It uses a combination of preprocessing, feature extraction, and classification algorithms to make predictions with a confidence score.

The system also handles **uncertain predictions**, making it more realistic and reliable compared to basic classifiers.

---

## 🚀 Key Features
- 🔍 Detects Fake vs Real News
- 🧠 NLP-based text preprocessing
- 📊 TF-IDF vectorization with n-grams
- ⚡ Multiple ML models (Logistic Regression & Passive Aggressive)
- 🎯 Confidence-based prediction system:
  - ✅ REAL  
  - ❌ FAKE  
  - ⚠️ UNCERTAIN  
- 🌐 Interactive UI built using Streamlit

---

## 🧠 Machine Learning Pipeline
1. Data Collection (Fake + Real News datasets)
2. Text Cleaning (lowercase, remove noise)
3. Feature Extraction using TF-IDF
4. Model Training (Logistic Regression / Passive Aggressive)
5. Evaluation using Accuracy, Precision, Recall, F1-score
6. Deployment via Streamlit UI

---

## 📊 Dataset
Due to GitHub size limitations, datasets and model files are hosted externally:

👉 **Download Dataset & Model Files:**  
https://drive.google.com/drive/folders/1EOiG5r0PQpe6Q3p0jCI12lTZAjSY-w1P?usp=drive_link

Datasets used:
- Fake & Real News Dataset  


---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/priyanshjain2907/Fake-News-Detection.git
cd Fake-News-Detector
