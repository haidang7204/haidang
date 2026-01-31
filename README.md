# Porto Seguro’s Safe Driver Prediction

## Project Overview

This project aims to predict the probability that a driver will file an insurance claim in the next year. The dataset is highly imbalanced, with only 3.6% of drivers actually filing claims, making it a challenging classification problem where standard accuracy is not a reliable metric.

## 1. Project structure

- `insurance_risk_pipeline.ipynb`: main Colab-style notebook with EDA, feature engineering, modeling, and results.
- `preprocessing.py`: reusable Python module defining the `InsurancePreprocessor` class.
- `test_preprocessing.py`: pytest unit tests for the preprocessing pipeline.
- `requirements.txt`: Python package dependencies.
- `technical_report.pdf`: 3-5 page technical report.
- `slides_proj1.pdf`: 5 slide executive summary deck.
- `feynman_study_guide.pdf`: answers to deep dive questions.

## 2. Data Description
- The features are grouped into several categories based on their prefixes:

- ps_ind: Individual driver characteristics 

- ps_reg: Registration/Geographic information.

- ps_car: Vehicle-related features.

- ps_calc: Calculated features provided by the company.

## 3.  How to reproduce the notebook results
- Open insurance_risk_pipeline.ipynb in Jupyter / VS Code / Colab.

- Make sure the environment has the packages from requirements.txt installed.

- Run all cells from top to bottom:

- Data loading and EDA (Tasks 1–3)

- Feature engineering and selection (Tasks 4–5)

- Preprocessing pipeline with InsurancePreprocessor (Task 6)

- Model training, evaluation, and plots (Task 7)

+ At the end you should see the validation metrics:

- Logistic Regression: ROC‑AUC ≈ 0.609

- Random Forest: ROC‑AUC ≈ 0.555

- XGBoost: ROC‑AUC ≈ 0.613 (best model)

## 4. How to run the project on new data
Import the preprocessor and the trained model (or retrain the model using the notebook).

Apply the fitted InsurancePreprocessor to new driver records.

Use the XGBoost model to generate predicted claim probabilities.

Rank drivers by probability and flag the top 10% as highest risk.


