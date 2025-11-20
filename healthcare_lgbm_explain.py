"""
healthcare_lgbm_explain.py
End-to-end LightGBM demo for healthcare predictions (readmission risk).
- Synthetic clinical dataset generation
- Feature engineering
- Train LightGBM for binary readmission prediction
- Evaluate & save metrics
- Produce SHAP (global + local), LIME (local) and PDP artifacts
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from lime import lime_tabular

RND = 42
np.random.seed(RND)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def simulate_healthcare_data(n=50000, readmit_rate=0.12):
    """
    Simulate a semi-realistic patient dataset.
    - n: number of patient visits / discharges
    - readmit_rate: baseline probability of 30-day readmission
    """
    # Demographics
    age = np.random.randint(18, 95, size=n)
    sex = np.random.choice([0,1], size=n, p=[0.52, 0.48])  # 0:F, 1:M
    bmi = np.clip(np.random.normal(27, 6, size=n), 15, 55)

    # Visit features
    length_of_stay = np.clip(np.random.exponential(4, size=n).astype(int) + 1, 1, 60)
    num_prev_adm = np.random.poisson(1.2, size=n)
    days_since_last_adm = np.random.exponential(60, size=n).astype(int)

    # Vitals & labs synthetic signals
    systolic_bp = np.clip(np.random.normal(125, 18, size=n) - (age-60)*0.2, 70, 240)
    heart_rate = np.clip(np.random.normal(80, 12, size=n), 40, 180)
    temp_c = np.clip(np.random.normal(36.7, 0.6, size=n), 34, 41)
    wbc = np.clip(np.random.normal(7.0, 3.0, size=n), 1, 40)  # 10^9/L
    creatinine = np.clip(np.random.normal(1.1, 0.6, size=n), 0.3, 15.0)
    hemoglobin = np.clip(np.random.normal(13.0, 2.0, size=n), 6, 19)

    # Comorbidity flags (binary)
    diabetes = np.random.binomial(1, 0.18, size=n)
    hf = np.random.binomial(1, 0.08, size=n)  # heart failure
    copd = np.random.binomial(1, 0.05, size=n)
    cancer = np.random.binomial(1, 0.06, size=n)
    stroke = np.random.binomial(1, 0.03, size=n)

    # Social factors
    smoking = np.random.binomial(1, 0.20, size=n)
    living_alone = np.random.binomial(1, 0.22, size=n)
    followup_scheduled = np.random.binomial(1, 0.65, size=n)

    # Medication complexity
    num_medications = np.clip(np.random.poisson(4, size=n) + (diabetes*1) + (hf*2), 0, 30)

    # Baseline risk score (latent)
    base_score = (
        0.02 * (age - 50) +
        0.04 * length_of_stay +
        0.5 * num_prev_adm +
        0.4 * diabetes +
        0.8 * hf +
        0.5 * copd +
        0.05 * (num_medications / 5) +
        -0.6 * followup_scheduled +
        0.003 * (wbc - 7) +
        0.3 * (creatinine > 2.0).astype(int)
    )
    # Map to probability with logistic transform, scaled around readmit_rate
    logits = base_score - np.percentile(base_score, 50)  # center
    probs = 1 / (1 + np.exp(-logits))
    # calibrate to desired population rate
    probs = probs * (readmit_rate / probs.mean())

    # synthetic continuous risk score (0-1)
    risk_score = np.clip(probs, 0, 1)

    # Binary outcome sampled from probability
    readmit = np.random.binomial(1, risk_score)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "length_of_stay": length_of_stay,
        "num_prev_adm": num_prev_adm,
        "days_since_last_adm": days_since_last_adm,
        "systolic_bp": systolic_bp,
        "heart_rate": heart_rate,
        "temp_c": temp_c,
        "wbc": wbc,
        "creatinine": creatinine,
        "hemoglobin": hemoglobin,
        "diabetes": diabetes,
        "heart_failure": hf,
        "copd": copd,
        "cancer": cancer,
        "stroke": stroke,
        "smoking": smoking,
        "living_alone": living_alone,
        "followup_scheduled": followup_scheduled,
        "num_medications": num_medications,
        "readmit_prob_synthetic": risk_score,
        "readmitted": readmit
    })

    return df

def feature_engineering(df):
    # Typical engineered features
    df["age_bucket"] = pd.cut(df["age"], bins=[17,40,60,75,100], labels=[0,1,2,3]).astype(int)
    df["high_bmi_flag"] = (df["bmi"] >= 30).astype(int)
    df["anemia_flag"] = (df["hemoglobin"] < 11).astype(int)
    df["renal_failure_flag"] = (df["creatinine"] > 2.0).astype(int)
    df["abnormal_vitals"] = ((df["systolic_bp"] < 90) | (df["temp_c"] > 38) | (df["heart_rate"] > 120)).astype(int)

    # Interaction features
    df["hf_and_diabetes"] = (df["heart_failure"] & df["diabetes"]).astype(int)
    df["recent_adm_flag"] = (df["days_since_last_adm"] < 30).astype(int)
    df["meds_per_prev_adm"] = df["num_medications"] / (1 + df["num_prev_adm"])

    # drop columns not used or keep as is
    # Keep readmit_prob_synthetic for diagnostics only (not used in model)
    return df

def train_model(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,
        "verbosity": -1,
        "seed": RND
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=1000, early_stopping_rounds=40, verbose_eval=50)
    return model

def evaluate(model, X_test, y_test):
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
    print("Confusion Matrix:\n", cm)
    return probs

def shap_and_pdp(model, X_train, X_test, sample_idx=7):
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot (global)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    shap_summary_file = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(shap_summary_file, dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot to {shap_summary_file}")

    # SHAP waterfall for a sample (local)
    try:
        sw_file = os.path.join(OUT_DIR, f"shap_waterfall_sample{sample_idx}.png")
        plt.figure(figsize=(8,4))
        shap.plots.waterfall(shap.Explanation(values=shap_values[sample_idx],
                                             base_values=explainer.expected_value,
                                             data=X_test.iloc[sample_idx]), show=False)
        plt.tight_layout()
        plt.savefig(sw_file, dpi=150)
        plt.close()
        print(f"Saved SHAP waterfall to {sw_file}")
    except Exception as e:
        print("Could not save SHAP waterfall:", e)

    # PDP for length_of_stay & num_prev_adm
    for feat in ["length_of_stay", "num_prev_adm"]:
        if feat in X_train.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            PartialDependenceDisplay.from_estimator(model, X_train, [feat], ax=ax)
            pdp_file = os.path.join(OUT_DIR, f"pdp_{feat}.png")
            plt.tight_layout()
            plt.savefig(pdp_file, dpi=150)
            plt.close()
            print(f"Saved PDP for {feat} to {pdp_file}")

def lime_explain(model, X_train, X_test, idx=5):
    feature_names = X_train.columns.tolist()
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode="classification",
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
    print(f"Saved LIME explanation to {html_file}")

def main():
    print("Simulating healthcare dataset...")
    df = simulate_healthcare_data(n=40000, readmit_rate=0.12)
    print("Dataset simulated:", df.shape)
    df = feature_engineering(df)
    print("Features engineered. Columns:", df.shape[1])

    # Prepare train/val/test splits
    target = "readmitted"
    X = df.drop(columns=["readmitted", "readmit_prob_synthetic"])  # remove synthetic prob
    y = df[target].astype(int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RND)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RND)
    print("Split sizes (train/val/test):", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # Train model
    print("Training LightGBM model...")
    model = train_model(X_train, y_train, X_val, y_val)
    joblib.dump(model, os.path.join(OUT_DIR, "lgbm_readmit_model.pkl"))
    print("Saved model to outputs/lgbm_readmit_model.pkl")

    # Evaluate
    print("Evaluating model...")
    probs = evaluate(model, X_test, y_test)

    # Explainability
    print("Generating SHAP, PDP artifacts...")
    shap_and_pdp(model, X_train, X_test, sample_idx=10)
    print("Generating LIME explanation...")
    lime_explain(model, X_train, X_test, idx=10)

    print("Done. Check outputs/ for artifacts.")

if __name__ == "__main__":
    main()
