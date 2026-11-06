# Waze User Churn Prediction & Analysis

A professional end-to-end data science and machine learning project predicting user churn for Waze. This repository showcases rigorous data cleaning, exploratory data analysis (EDA), feature engineering, and comparative modeling using a baseline **Logistic Regression** model and an ensemble **Random Forest** classifier.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Data Insights & EDA](#key-data-insights--eda)
3. [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
4. [Machine Learning Models & Evaluation](#machine-learning-models--evaluation)
5. [Feature Importance & Interpretability](#feature-importance--interpretability)
6. [Strategic Business Recommendations](#strategic-business-recommendations)
7. [Repository Structure](#repository-structure)

---

## 🎯 1. Project Overview
User retention is critical for maintaining Waze’s traffic reporting density and core platform utility. This project analyzes user engagement behavior to predict which drivers are at high risk of churning, allowing product and data teams to deploy targeted retention strategies.

* **Dataset:** 14,999 records containing user session metrics, driving frequencies, distances, tenures, and device types.
* **Class Balance:** Moderate class imbalance identified (~82.3% retained users vs. ~17.7% churned users).
* **Primary Evaluation Metric:** **ROC-AUC** (to ensure robust performance across all classification thresholds despite class imbalance) alongside precision, recall, and F1-scores.

---

## 📊 2. Key Data Insights & EDA
Through exploratory data analysis and missing value investigations (validated as Missing Completely at Random - MCAR):
* **Compressed Activity Windows:** Churned users actually logged higher volumetric intensity in their final active month (higher medians in sessions, total drives, and distance driven) but packed this usage into a significantly compressed timeframe (`activity_days` median of 8.0 vs. 17.0 for retained users). This indicates erratic, high-intensity burnout before churn.
* **Driver Type Disparity:** Non-professional drivers experience a significantly higher churn rate (**19.9%**) compared to professional drivers (**7.6%**), making non-professionals ~2.6× more likely to abandon the app.
* **Data Integrity Audit:** An audit of travel speed ratios revealed that 37% of records exceeded realistic physical constraints (>150 km/h) due to synthetic data generation artifacts; these derived ratios were successfully excluded from modeling to prevent noise injection.

---

## ⚙️ 3. Feature Engineering & Preprocessing
To optimize data for model ingestion:
* **Log Transformations:** Applied `np.log1p` to highly right-skewed core metrics (`sessions`, `drives`, `km_driven`, `duration_mins`, `fav_nav1`, `fav_nav2`, `total_sessions`) for linear modeling.
* **Custom Features:** Engineered `professional_driver` flags based on high-frequency trip thresholds, alongside travel ratios (`km_per_drive`, `km_per_driving_day`, `drives_per_driving_day`).
* **Categorical Encoding:** One-hot encoded device types (`device_iPhone`) using pandas dummy variables.
* **Data Scaling:** Applied `StandardScaler` exclusively to training data (`X_train_log`) to prevent data leakage.

---

## 🤖 4. Machine Learning Models & Evaluation
Two distinct model architectures were trained and evaluated using clean stratified 80/20 train/test splits with `class_weight='balanced'` adjustments:

1. **Baseline Model — Logistic Regression (Scaled Log Features)**
   * **ROC-AUC Score:** `0.7230`
   * **Recall (Churned Class):** `0.69`
   * *Summary:* Highly sensitive at capturing at-risk users, making it an effective broad-net risk detection tool.

2. **Ensemble Model — Random Forest Classifier (Raw Ensemble Features)**
   * **ROC-AUC Score:** `0.7071`
   * **Accuracy:** `0.79`
   * *Summary:* Cleaner at avoiding false alarms among retained users, achieving higher overall precision and specificity.

---

## 📈 5. Feature Importance & Interpretability
* **Logistic Regression Coefficients:** Identified total monthly trip volume (`drives_log`), primary favorite location usage (`fav_nav1_log`), and trip duration (`duration_mins_log`) as primary positive linear drivers of churn, while activity consistency (`activity_days`, `driving_days`) strongly protected against churn.
* **Random Forest Importances:** Highlighted `activity_days`, `km_per_driving_day`, and `driving_days` as top non-linear decision boundaries, proving that temporal pacing and consistency dictate retention.

---

## 💡 6. Strategic Business Recommendations
Based on empirical model findings, Waze can deploy the following actions:
* **Targeted Interventions for Non-Professionals:** Direct retention and onboarding campaigns primarily toward non-professional drivers to curb the core driver of platform attrition.
* **Early-Warning Automated Triggers:** Monitor users whose active days narrow while daily trip volume temporarily spikes. Use automated push notifications or alternative routing check-ins as early-warning triggers to re-stabilize engagement before burnout occurs.

---

## 📁 7. Repository Structure
```text
├── waze_dataset.csv              # Raw input data
├── Waze-Capstone-v3.ipynb        # Complete Jupyter Notebook (EDA, Modeling, Evaluation)
└── README.md                     # Project documentation
