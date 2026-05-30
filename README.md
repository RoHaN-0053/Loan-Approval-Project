# Loan-Approval-Project
End-to-end ML pipeline to predict loan approvals using classification algorithms, feature engineering, and precision-focused model evaluation.


# 💳 CreditWise — Loan Approval Prediction

A machine learning project that predicts whether a loan application 
will be approved based on applicant financial and demographic data.
Built with a focus on Precision over Accuracy — because in credit 
risk, approving a bad loan is far more costly than rejecting a good one.

---

## 📌 Problem Statement
Banks and financial institutions process thousands of loan applications
daily. This project automates the approval prediction process using 
historical applicant data, helping identify creditworthy applicants 
more reliably.

---

## 📂 Dataset Features
- Applicant Income
- Credit Score
- Debt-to-Income (DTI) Ratio
- Employment Status
- Loan Purpose
- Property Area
- Gender, Marital Status, Education Level

---

## ⚙️ Project Pipeline
1. **Data Cleaning** — Mean/mode imputation for missing values
2. **EDA** — Distribution plots, income vs credit score analysis,
             employment vs DTI visualizations
3. **Encoding** — Label Encoding + One-Hot Encoding for
                  categorical features
4. **Feature Engineering** — Log transform on income, polynomial
                              features on DTI and Credit Score
5. **Scaling** — StandardScaler for all numerical features
6. **Modeling** — Three classifiers trained and evaluated
7. **Evaluation** — Precision, Recall, F1, Accuracy + Confusion Matrix

---

## 🤖 Models Used
| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| K-Nearest Neighbors | Distance-based classifier |
| Gaussian Naive Bayes | Best Precision — selected as final model |

---



## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 💡 Key Takeaway
A balanced dataset was confirmed before modeling, removing the need
for resampling techniques. Naive Bayes was selected as the best model
based on Precision — the most critical metric in a loan approval context.

---

## 🚀 How to Run
1. Clone the repo
   git clone https://github.com/RoHaN-0053/creditwise-loan-prediction
2. Install dependencies
   pip install -r requirements.txt
3. Open the notebook
   jupyter notebook Creditwise_loan.ipynb
