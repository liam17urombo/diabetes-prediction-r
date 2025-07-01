# 🧠 Diabetes Prediction Using Machine Learning in R

This project uses the **Pima Indians Diabetes dataset** to build predictive models that estimate the risk of diabetes based on clinical and personal health indicators.

The goal is to compare simple and complex machine learning models — like **Naive Bayes**, **XGBoost**, and **LightGBM** — and identify which performs best in terms of:
- ✅ High AUC (Area Under the Curve)
- ✅ Minimal overfitting (small gap between training and testing accuracy)

---

## 📊 Dataset

- 768 rows × 9 columns
- Source: [Pima Indians Diabetes Dataset (UCI)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features include:
  - Glucose, BloodPressure, BMI, Age, Insulin, etc.
  - Outcome (0 = No diabetes, 1 = Diabetes)

---

## 🔧 Methods & Tools

- **R & RStudio**
- **tidymodels**, **tidyverse** for modeling
- **DALEX** for model explainability
- **themis** for SMOTE oversampling
- Models trained:
  - Naive Bayes (`discrim`, `klaR`)
  - XGBoost (`parsnip` + `xgboost`)
  - LightGBM (`bonsai`)

---

## 🚀 How to Run the Project

1. Clone or download the repo
2. Open `diabetes_prediction.R` or `.Rmd` file in RStudio
3. Install required packages (first time only):
   ```r
   install.packages(c("tidymodels", "themis", "DALEX", "discrim", "xgboost", "bonsai"))
