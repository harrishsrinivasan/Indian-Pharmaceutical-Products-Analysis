#  Indian Pharmaceutical Products Analysis

Predict drug prices and identify discontinued products using machine learning on Indian pharma data.

---

##  Objectives

- Analyze features like dosage form, strength, pack size, and manufacturer
- Predict `price_inr` using regression
- Predict `is_discontinued` using classification
- Handle imbalance with SMOTE
- Evaluate using R², RMSE, Precision, Recall, and F1-Score

---

##  Dataset Overview

Key Features:
- `brand_name`, `manufacturer_raw`, `dosage_form`
- `pack_size`, `pack_unit`, `strength_mg`, `therapeutic_class`
- `num_active_ingredients`, `price_inr`, `is_discontinued`

Engineered Feature:
- `price_per_unit` = `price_inr / pack_size`

Filtered outliers using the **99th percentile** of `price_inr` and `price_per_unit`.

---

##  ML Models

###  Classification

| Model              | Accuracy | Precision | Recall | F1-score | Best Use Case |
|-------------------|----------|-----------|--------|----------|----------------|
| Random Forest      | 0.97     | 0.73      | 0.07   | 0.13     | Minimize false alarms |
| XGBoost + SMOTE    | 0.92     | 0.21      | 0.55   | 0.31     | Catch more discontinued products |

<details>
<summary>🔍 Why This Happened</summary>

- `is_discontinued` is highly imbalanced
- **Random Forest** predicted mostly "0", boosting accuracy but missing true positives
- **XGBoost + SMOTE** improved recall by generating synthetic minority data, trading off some precision
</details>

---

###  Regression

- **Model**: `RandomForestRegressor`
- **R² Score**: `0.9995`
- **RMSE**: ₹4.54

<details>
<summary> Why It Worked So Well</summary>

- Outliers removed using 99th percentile
- `np.log1p(price_inr)` stabilized variance
- Strong predictors like `price_per_unit`, `pack_size`, and `strength_mg`
- Random Forest captured nonlinear patterns and handled categorical features well
</details>

---
