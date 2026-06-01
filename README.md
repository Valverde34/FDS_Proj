# Pediatric Cardiac Disease Prediction

Machine learning pipeline for predicting cardiac abnormalities in children, built on a real-world clinical dataset (UCMF) from a Portuguese pediatric cardiology unit.

Built as a final project for the **Fundamentals of Data Science** course at Universidade de Aveiro (2024/2025).

---

## Overview

The goal of this project is to predict whether a child presents a normal or abnormal cardiac condition based on clinical measurements. The full pipeline covers everything from raw data exploration to predictive modelling, producing 50+ outputs including statistical reports, visualisations, and model evaluations.

---

## Dataset

The UCMF dataset contains anonymised pediatric cardiology records with the following features:

| Feature | Description |
|---|---|
| IDADE | Patient age (0-19 years) |
| SEXO | Sex |
| IMC | Body Mass Index |
| PA SISTOLICA | Systolic blood pressure |
| PA DIASTOLICA | Diastolic blood pressure |
| FC | Heart rate |
| PULSOS | Pulse assessment |
| SOPRO | Heart murmur |
| B2 | Second heart sound |
| NORMAL X ANORMAL | Target variable (Normal / Abnormal) |

---

## Pipeline

### 1. Raw Data Analysis — `raw_data_analysis.py`

Exploratory analysis of the raw dataset before any preprocessing.

**Key outputs:** variable classification, missing value analysis, physiologically impossible values, descriptive statistics, histograms, boxplots, KDE density curves, categorical distributions.

### 2. Data Preprocessing — `pre_process.py`

Cleaning and preparation for downstream analysis.

**Transformations applied:**
- Duplicate removal (ID column)
- Age validation (0-19 years)
- Weight, height and BMI cleaning
- Systolic/diastolic blood pressure and heart rate validation
- Categorical standardisation (SEXO, TARGET, PULSOS, SOPRO, B2)
- Irrelevant column removal

**Output:** `UCMF_CLEAN.xlsx`

### 3. Exploratory Data Analysis — `eda_basic.py`

Detailed descriptive analysis of the cleaned dataset: statistics summary, histograms, boxplots, dispersion measures, normality tests (Shapiro-Wilk, Kolmogorov-Smirnov).

### 4. Bivariate Analysis — `bivariate_analysis.py`

Analysis of pairwise variable relationships: Pearson, Spearman and Kendall correlations, simple linear regressions, t-tests comparing Normal vs Abnormal groups, ANOVA for categorical variables, partial correlations controlling for age.

### 5. Multivariate Analysis — `multivariate_analysis.py`

Complex multi-variable relationships: logistic regression with Odds Ratios, Mutual Information scores, PCA (scree plot, biplot, loadings), K-Means clustering with Silhouette and Davies-Bouldin evaluation.

### 6. Feature Selection — `feature_selection.py`

Identification of the most relevant predictive variables using: F-test, Mutual Information, Random Forest importance, Gradient Boosting importance, Recursive Feature Elimination (RFE), cross-validation performance curve.

### 7. Predictive Modelling — `predictive_models.py`

Training and evaluation of 5 classification models:

| Model | Description |
|---|---|
| Decision Tree | Interpretable tree-based classifier |
| SVM | Support Vector Machine with RBF kernel |
| Naive Bayes | Gaussian probabilistic classifier |
| Random Forest | Ensemble of decision trees |
| Neural Network | Multi-layer Perceptron (MLP) |

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, 5-fold cross-validation.

---

## How to run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl xlrd
```

### Run the full pipeline in order

```bash
python src/raw_data_analysis.py
python src/pre_process.py
python src/eda_basic.py
python src/bivariate_analysis.py
python src/multivariate_analysis.py
python src/feature_selection.py
python src/predictive_models.py
```

Each script reads from the previous step's output. Start with `raw_data_analysis.py` and run sequentially.

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=flat&logo=python&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-3776AB?style=flat&logo=python&logoColor=white)

