# %% [markdown]
# ### Setup & Imports

# %%
# Data handling
import pandas as pd
import numpy as np
import yaml
import joblib
import time
import matplotlib.pyplot as plt

# Models
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb

# MLflow
import mlflow
import mlflow.sklearn

# Warnings
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ### Load Config

# %%
# Load YAML config
with open("configs/training.yml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# %% [markdown]
# ### Load Preprocessed Data

# %%
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_val = pd.read_csv("data/processed/X_val.csv")
y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)

# %% [markdown]
# ### Define Models & Hyperparameters

# %%
# LightGBM
lgb_config = next(m for m in config['models'] if m['type'] == 'lightgbm')
lgb_model = lgb.LGBMClassifier(
    max_depth=lgb_config['max_depth'],
    learning_rate=lgb_config['learning_rate'],
    n_estimators=lgb_config['n_estimators']
)
lgb_params = config['hyperparameter_search']['lightgbm']

# %%
# XGBoost
xgb_config = next(m for m in config['models'] if m['type'] == 'xgboost')
xgb_model = xgb.XGBClassifier(
    max_depth=xgb_config['max_depth'],
    learning_rate=xgb_config['learning_rate'],
    n_estimators=xgb_config['n_estimators'],
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_params = config['hyperparameter_search']['xgboost']

# %% [markdown]
# ### Initialize MLflow

# %%
mlflow.set_experiment(config['logging']['experiment'])

# %% [markdown]
# ### Train and Log LightGBM

# %%
with mlflow.start_run(run_name="LightGBM"):
    start = time.time()
    
    # GridSearchCV for hyperparameter tuning
    lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='roc_auc', n_jobs=-1) # type: ignore
    lgb_grid.fit(X_train, y_train)
    
    # Predict on validation
    y_val_pred = lgb_grid.predict(X_val)
    y_val_proba = lgb_grid.predict_proba(X_val)[:,1]
    
    # Metrics
    roc_auc = roc_auc_score(y_val, lgb_grid.predict_proba(X_val), multi_class='ovr')
    f1 = f1_score(y_val, y_val_pred, average='macro')
    auc = accuracy_score(y_val, y_val_pred)
    end = time.time()
    
    # Log
    mlflow.log_params(lgb_grid.best_params_)
    mlflow.log_metric("roc_auc", roc_auc) # type: ignore
    mlflow.log_metric("f1_score", f1) # type: ignore
    mlflow.log_metric("train_time_sec", end-start)
    mlflow.sklearn.log_model(lgb_grid.best_estimator_, "model") # type: ignore
    
    print(f"LightGBM - ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}, auc : {auc:.4}")

# %% [markdown]
# ### Train and Log XGBoost

# %%
with mlflow.start_run(run_name="XGBoost"):
    start = time.time()
    
    # GridSearchCV for hyperparameter tuning
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    
    # Predict on validation
    y_val_pred = xgb_grid.predict(X_val)
    y_val_proba = xgb_grid.predict_proba(X_val)[:,1]
    
    # Metrics
    roc_auc = roc_auc_score(y_val, xgb_grid.predict_proba(X_val), multi_class='ovr')
    f1 = f1_score(y_val, y_val_pred, average='macro')
    auc = accuracy_score(y_val, y_val_pred)
    end = time.time()
    
    # Log
    mlflow.log_params(xgb_grid.best_params_)
    mlflow.log_metric("roc_auc", roc_auc) # type: ignore
    mlflow.log_metric("f1_score", f1) # type: ignore
    mlflow.log_metric("train_time_sec", end-start)
    mlflow.sklearn.log_model(xgb_grid.best_estimator_, "model") # type: ignore
    
    print(f"XGBoost - ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}, auc : {auc:.4}")


# %% [markdown]
# ### Compare Models & Select Best|

# %%
# Example: Compare ROC-AUC on validation
results = {
    "LightGBM": roc_auc_score(y_val, lgb_grid.predict_proba(X_val), multi_class='ovr'),
    "XGBoost": roc_auc_score(y_val, xgb_grid.predict_proba(X_val), multi_class='ovr')
}

best_model_name = max(results, key=results.get    ) # type: ignore
best_model = lgb_grid.best_estimator_ if best_model_name=="LightGBM" else xgb_grid.best_estimator_

print(f"Best Model: {best_model_name} with ROC-AUC = {results[best_model_name]:.4f}")

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

# %% [markdown]
# ### Learning Curve

# %%
# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve( # type: ignore
    xgb_grid, X_train, y_train,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# %%
# Compute mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1) 

# %%
# Plot
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validation score")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, color="r", alpha=0.1)
plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, color="g", alpha=0.1)
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/learning_curve.png", bbox_inches="tight")
plt.show()

# %%
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
best_model1 = joblib.load("models/best_model.pkl")
y_test_pred = best_model1.predict(X_test) # type: ignore

auc = accuracy_score(y_test_pred, y_test)
auc

# %%



