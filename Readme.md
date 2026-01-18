# 🛡️ Cyberbullying Detection & Severity Classification Using Machine Learning and Deep Learning

This project implements an end-to-end **Cyberbullying Detection and Severity Classification system** using a combination of **traditional Machine Learning**, **Deep Learning (LSTM)**, **Transformer-based models (BERT)**, and a **Hybrid Ensemble Voting Classifier**.

The notebook is developed and executed in **Google Colab** as part of an academic research project / MSc Data Science thesis.

---

## 📌 Problem Statement

Cyberbullying on social media platforms has become a critical issue affecting mental health and online safety.  
This project focuses on:

- **Detecting cyberbullying content**
- **Classifying the severity level** of cyberbullying (e.g., Low, Medium, High)
- **Comparing ML, DL, and Transformer approaches**
- **Improving accuracy using ensemble learning**

---

## 🧠 Solution Overview

The notebook follows a structured ML lifecycle:

1. **Data Loading & Exploration**
2. **Text Preprocessing & Cleaning**
3. **Feature Engineering**
4. **Model Training**
5. **Model Evaluation**
6. **Ensemble Learning**
7. **Model Persistence**

---

## 🏗️ Architecture & Models Used

### 🔹 Traditional Machine Learning
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- XGBoost

### 🔹 Deep Learning
- LSTM (Keras / TensorFlow)

### 🔹 Transformer Model
- BERT (Sentence Transformers)

### 🔹 Ensemble Learning
- **Hybrid Voting Classifier**
  - Random Forest
  - SVM
  - XGBoost

---

## 📂 Dataset

- **Source**: Cyberbullying Tweets dataset from Kaggle
- **Format**: CSV
- **Text Field**: Tweet content
- **Target**:
  - Binary Classification (Bullying / Non-Bullying)
  - Severity Classification

The dataset is loaded from **Google Drive**.

---

## ⚙️ Environment & Dependencies

The notebook runs entirely on **Google Colab**.

### Key Libraries:
- `pandas`, `numpy`
- `scikit-learn`
- `tensorflow`, `keras`
- `xgboost`
- `sentence-transformers`
- `matplotlib`, `seaborn`
- `nltk`

Dependencies are installed dynamically inside the notebook.

---

## ▶️ How to Run

1. Open the notebook in **Google Colab**
2. Mount Google Drive when prompted
3. Create a token from HuggingFace to be used.
4. Ensure dataset paths are correctly configured
5. Run cells sequentially from top to bottom
6. Review evaluation metrics and visualizations

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Each model’s performance is compared to identify the best approach.

---

## 💾 Model Saving

- Trained models are saved to **Google Drive**
- Tokenizers and embeddings are persisted for reuse
- Supports reproducibility and further experimentation

---

## 📈 Key Outcomes

- Transformer-based models outperform traditional ML in complex language understanding
- Ensemble models improve robustness and overall accuracy
- Severity classification provides deeper insight beyond binary detection

---

## 📌 Use Cases

- Social media moderation
- Online safety platforms
- Educational research
- Abuse detection systems

---

## 🚀 Future Enhancements

- Multi-lingual cyberbullying detection
- Real-time streaming inference
- Explainable AI (SHAP / LIME)
- Deployment via REST API

---

## 👤 Author

**Rahul Sharma**  
MSc Data Science  
Liverpool John Moores University  

---

## 📄 License

This project is intended for **academic and research purposes**.  
Reuse with proper attribution.

