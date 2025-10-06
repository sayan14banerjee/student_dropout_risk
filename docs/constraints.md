# Project Constraints: Dropout Risk Prediction

This document defines the **non-functional and technical constraints** for the Dropout Risk Prediction project to ensure reproducibility, performance, and maintainability.

---

## 1. **Dataset Constraints**
- Maximum dataset size: **100,000 rows** (to ensure fast training and evaluation).  
- Dataset format: **CSV** with clearly defined headers.  
- Features should be **numerical or properly encoded categorical variables**.  
- Target variable: `dropout_risk` (binary: 0 → No Risk, 1 → At Risk).  

---

## 2. **Model Constraints**
- Only **interpretable models** are allowed (e.g., RandomForest, XGBoost, Logistic Regression).  
- Maximum training time per run: **30 minutes** on a standard machine (8GB RAM, 4-core CPU).  
- Model size should **not exceed 200 MB** for storage and deployment efficiency.  

---

## 3. **Training & Experiment Constraints**
- Hyperparameters and metrics must be **logged automatically using MLflow**.  
- Metrics to track per run:  
  - **ROC-AUC**  
  - **F1 Score**  
  - Accuracy (optional)  
- Only one **validation set** is allowed; no data leakage from the test set.  

---

## 4. **Code & Infrastructure Constraints**
- Code must be **modular**, separated into `src/` scripts:  
  - `data_preprocessing.py`  
  - `train_model.py`  
  - `evaluate_model.py`  
- Configuration parameters (hyperparameters, paths) must be stored in **`configs/training_config.yaml`**.  
- All experiments should be **reproducible** using fixed random seeds.  

---

## 5. **Deployment & Reproducibility Constraints**
- Final model should be **exported using MLflow** or `joblib` for future inference.  
- Inference script must accept **new input data in CSV format**.  
- Training and evaluation results must be **reproducible on any machine** with the same environment.  

---

## 6. **Documentation Constraints**
- All changes must be **tracked in Git**.  
- `docs/constraints.md` should always reflect the latest project rules.  
- High-level project overview, setup instructions, and experiment summaries should be maintained in **GitHub Wiki**.

---

> ⚠️ Note: Adhering to these constraints ensures a **robust, maintainable, and professional ML project** that is reproducible, interpretable, and easy to deploy.
