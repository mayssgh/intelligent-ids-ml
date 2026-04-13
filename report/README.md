# 🔐 Intelligent Intrusion Detection System (IDS)

## 📌 Overview
This project implements a **Machine Learning-based Intrusion Detection System (IDS)** designed to detect malicious network traffic using the CICIDS2017 dataset.

It combines data preprocessing, model training, explainable AI, and deployment into a complete intelligent system.

---

## 🚀 Features
- 🔧 Data preprocessing pipeline (cleaning, encoding, scaling)
- 🤖 Multiple ML models:
  - Logistic Regression
  - Random Forest (best performing)
  - Multi-Layer Perceptron (MLP)
- 📊 Model evaluation (Accuracy, Precision, Recall, F1-score)
- 🔍 Explainable AI using SHAP
- 🌐 REST API deployment with FastAPI

---

## 🧠 Best Model Performance

| Model               | Accuracy | F1-score |
|--------------------|----------|----------|
| Logistic Regression| 0.9867   | 0.9866   |
| Random Forest      | **0.9985** | **0.9985** |
| MLP                | 0.9948   | 0.9948   |

👉 Random Forest achieved the best performance due to its ability to capture complex patterns in network traffic.

---

## 📂 Project Structure
intelligent-ids-ml/
│
├── data/ # Dataset (ignored in Git)
├── models/ # Saved models
├── notebooks/ # Optional experiments
├── results/ # Outputs / metrics
│
├── src/
│ ├── preprocessing.py
│ ├── models.py
│ ├── evaluation.py
│ ├── explainability.py
│ ├── train.py
│ ├── api.py
│ └── predict.py
│
├── report/ # Final report
├── presentation/ # Slides
├── requirements.txt
└── README.md

---

## ⚙️ Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- SHAP (Explainable AI)
- FastAPI (API deployment)

---

## ▶️ How to Run

