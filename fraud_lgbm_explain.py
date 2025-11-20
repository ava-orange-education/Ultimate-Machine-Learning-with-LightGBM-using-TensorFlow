"""
fraud_lgbm_explain.py
End-to-end LightGBM fraud detection demo with SHAP, LIME and PDP
Generates synthetic imbalanced data, trains LightGBM, evaluates, and saves explainability artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import shap
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from lime import lime_tabular
import joblib
import warnings
warnings.filterwarnings("ignore")

RND = 42
np.random.seed(RND)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_synthetic_transactions(n_samples=200000, fraud_ratio=0.002):
    """
    Create synthetic transaction-like data to mimic fraud detection problems.
    n_samples - number of rows
    fraud_ratio - proportion of frauds (very small)
    """
    # Primary informative features using sklearn helper
    n_informative = 8
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.01,
        class_sep=1.0,
        random_state=RND
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["is_fraud"] = y

    # Create domain-like features
    # transaction_amount (positive skew)
    amounts = np.exp(df["f0"]) * 10
    df["transaction_amount"] = (amounts - amounts.min()) + np.random.gamma(1.2, 50, size=n_samples)

    # hour of day 0-23 (use feature projection)
    df["hour"] = (np.abs(df["f1"]) * 3).astype(int) % 24

    # merchant category (categorical)
    df["merchant_cat"] = pd.cut(df["f2"], bins=6, labels=[f"mc_{i}" for i in range(6)])
    # device novelty: randomly mark some devices as new (approx correlated with fraud)
    df["device_new"] = (df["f3"] + np.random.randn(n_samples) * 0.5) > 1.0
    df["device_new"] = df["device_new"].astype(int)

    # geo_distance proxy: distance from user's home (km)
    df["geo_distance_km"] = np.abs(df["f4"]) * 100

    # user historical avg amount (group-level proxy)
    # create fake user IDs and compute ave amount per user
    n_users = int(n_samples / 5)
    user_ids = np.random.randint(0, n_users, size=n_samples)
    df["user_id"] = user_ids
    df["user_avg_amount"] = df.groupby("user_id")["transaction_amount"].transform("mean")

    # transaction velocity: how many transactions in last 1h (synthetic)
    df["tx_velocity_1h"] = (np.abs(df["f5"]) * 5).astype(int)

    # drop raw numeric features used for simulation (optional)
    for c in [f"f{i}" for i in range(6)]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # ensure positive transaction amounts
    df["transaction_amount"] = df["transaction_amount"].clip(lower=1.0)

    # head
    return df

def feature_engineer(df):
    # Ratio of amount to user average
    df["amount_over_user_avg"] = df["transaction_amount"] / (df["user_avg_amount"] + 1e-6)

    # large_amount flag
    df["large_amount_flag"] = (df["transaction_amount"] > df["transaction_amount"].quantile(0.995)).astype(int)

    # unusual hour (1 if hour between 0-5)
    df["unusual_hour"] = df["hour"].isin([0,1,2,3,4,5]).astype(int)

    # merchant_cat one-hot
    df = pd.get_dummies(df, columns=["merchant_cat"], drop_first=True)

    # drop user_id (for modelling) but keep if needed for grouping later
    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])

    return df

def train_lightgbm(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        # helpful for skewed classes
        "is_unbalance": True,
        "seed": RND
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=500,
        early_stopping_rounds=30,
        verbose_eval=50
    )
    return model

def evaluate_and_save(model, X_test, y_test):
    probs = model.predict(X_test, num_iteration=model.best_iteration)
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"AUC: {auc:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    with open(os.path.join(OUT_DIR, "confusion_matrix.txt"), "w") as f:
        f.write(np.array2string(cm))

    print("=== EVALUATION SUMMARY ===")
    print(f"AUC: {auc:.6f}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return probs

def shap_and_pdp(model, X_train, X_test, sample_idx=1):
    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # summary plot saved
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    shap_summary_file = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(shap_summary_file, dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot to {shap_summary_file}")

    # force plot for a single instance
    idx = sample_idx
    try:
        force_plot_file = os.path.join(OUT_DIR, f"shap_force_sample{idx}.png")
        shap.initjs()
        # create force plot using shap, then save a matplotlib fallback
        shap_values_single = shap_values[idx]
        plt.figure(figsize=(8,3))
        shap.plots.waterfall(shap.Explanation(values=shap_values_single,
                                             base_values=explainer.expected_value,
                                             data=X_test.iloc[idx]), show=False)
        plt.tight_layout()
        plt.savefig(force_plot_file, dpi=150)
        plt.close()
        print(f"Saved SHAP waterfall for sample {idx} to {force_plot_file}")
    except Exception as e:
        print("Could not create SHAP force/waterfall plot:", e)

    # Partial Dependence Plot on transaction_amount (if exists)
    if "transaction_amount" in X_train.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        PartialDependenceDisplay.from_estimator(model, X_train, ["transaction_amount"], ax=ax)
        pdp_file = os.path.join(OUT_DIR, "pdp_transaction_amount.png")
        plt.tight_layout()
        plt.savefig(pdp_file, dpi=150)
        plt.close()
        print(f"Saved PDP for transaction_amount to {pdp_file}")
    else:
        print("transaction_amount not present in features for PDP.")

def lime_explain(model, X_train, X_test, idx=3):
    # LIME expects numpy arrays + training data for feature distributions
    feature_names = X_train.columns.tolist()
    class_names = ["legit", "fraud"]
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        mode="classification",
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=RND
    )

    exp = explainer.explain_instance(
        data_row=X_test.iloc[idx].values,
        predict_fn=lambda x: np.vstack([1 - model.predict(x), model.predict(x)]).T,
        num_features=8
    )

    html_file = os.path.join(OUT_DIR, "lime_explanation.html")
    exp.save_to_file(html_file)
    print(f"Saved LIME explanation for test idx {idx} to {html_file}")

def main():
    print("Generating synthetic dataset...")
    df = generate_synthetic_transactions(n_samples=100000, fraud_ratio=0.002)
    print("Initial dataset shape:", df.shape)
    df = feature_engineer(df)
    print("After feature engineering shape:", df.shape)

    target = "is_fraud"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # train/val/test split with stratify due to imbalance
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RND)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RND)

    print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # Train LightGBM
    print("Training LightGBM model...")
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    # Save model
    joblib.dump(model, os.path.join(OUT_DIR, "lgbm_model.pkl"))
    print("Saved LightGBM model to outputs/lgbm_model.pkl")

    # Evaluate
    print("Evaluating model...")
    probs = evaluate_and_save(model, X_test, y_test)

    # Explainability: SHAP, PDP
    print("Generating SHAP and PDP artifacts...")
    shap_and_pdp(model, X_train, X_test, sample_idx=5)

    # LIME explanation for a single example
    print("Generating LIME explanation...")
    lime_explain(model, X_train, X_test, idx=5)

    print("All done. Check the outputs/ directory for plots and artifacts.")

if __name__ == "__main__":
    main()
