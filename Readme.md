# Cyberbullying Severity Detection using Machine Learning

## Project Overview
This project focuses on **Cyberbullying Severity Detection** by analyzing social media data using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify online text into different levels of toxicity to help automate the identification of harmful content.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahul-sharma0507/Cyberbullying-Data-Analysis/blob/main/Rahul_CyberbullyingTweets.ipynb)

## 📌 Problem Statement

Cyberbullying on social media platforms has become a critical issue affecting mental health and online safety.  
This project focuses on:

- **Detecting cyberbullying content**
- **Classifying the severity level** of cyberbullying (e.g., Low, Medium, High)
- **Comparing ML, DL, and Transformer approaches**
- **Improving accuracy using ensemble learning**

## Key Features
- **Data Analysis:** Exploration of cyberbullying datasets to understand patterns in toxic behavior.
- **Severity Classification:** Multi-class classification to detect the intensity of cyberbullying (e.g., Ageism, Racism, Religion, etc.).
- **NLP Pipeline:** Text preprocessing including tokenization, stop-word removal, and lemmatization.
- **Model Training:** Implementation of advanced ML models using Python and Hugging Face.

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-Learn)
- **NLP Libraries** (NLTK, Transformers)
- **Google Colab** (for GPU-accelerated training)
- **Hugging Face Hub** (for pre-trained models)

## 📂 Dataset
The project utilizes a comprehensive dataset containing thousands of tweets labeled by the type and severity of cyberbullying.

- **Source**: Cyberbullying Tweets dataset from Kaggle
- **Format**: CSV
- **Text Field**: Tweet content
- **Target**:
  - Binary Classification (Bullying / Non-Bullying)
  - Severity Classification

The dataset is loaded from **Google Drive**.

## 🧠 Solution Overview

The notebook follows a structured ML lifecycle:

1. **Data Loading & Exploration**
2. **Text Preprocessing & Cleaning**
3. **Feature Engineering**
4. **Model Training**
5. **Model Evaluation**
6. **Ensemble Learning**
7. **Model Persistence**

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

## How to Run the Project
1. Click the **"Open in Colab"** button above to launch the notebook.
2. Add your Hugging Face API key to the Colab "Secrets" (Key icon) as `HF_TOKEN`.
3. Run all cells to see the data analysis and model results.

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


## 👤 Author

**Rahul Sharma**  
MSc Data Science  
Liverpool John Moores University  

## License
Distributed under the MIT License. See `LICENSE` for more information.
