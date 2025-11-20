#!/usr/bin/env python3
"""
ai_assisted_explainability.py

Full AI-Assisted Explainability bundle:
- simulate simple dataset
- train LightGBM regressor
- compute SHAP values (global + local)
- save SHAP summary plot, SHAP waterfall (local), top-features CSV
- generate a natural-language summary (template-based)
- OPTIONAL: if OPENAI_API_KEY env var exists and openai package installed, call OpenAI to get a richer summary

Usage:
    python ai_assisted_explainability.py

Outputs:
    explainability_bundle/
      - shap_summary.png
      - shap_waterfall_sample{N}.png
      - top_features.csv
      - nl_summary.txt
      - model.joblib
      - sample_predictions.csv
"""

import os
import sys
import json
import math
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

import shap
import matplotlib.pyplot as plt

# Optional LLM call (OpenAI). If you don't want this, just don't set OPENAI_API_KEY.
USE_OPENAI_IF_AVAILABLE = True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set this in your environment if you want LLM enhancement

def simulate_regression_data(n=20000, seed=42):
    """Simulate regression-style tabular data for demonstration."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.randint(18, 85, size=n),
        "income": np.round(rng.normal(50_000, 15_000, size=n)).astype(int),
        "num_prev_purchases": rng.poisson(2, size=n),
        "days_since_last_purchase": rng.exponential(30, size=n).astype(int),
        "is_promo_active": rng.binomial(1, 0.15, size=n),
        "item_price": np.round(rng.uniform(10, 300, size=n), 2)
    })
    # synthetic target with non-linear interactions
    base = (
        0.02 * (df["age"] - 40) +
        0.00002 * (df["income"] - 50_000) +
        0.5 * df["num_prev_purchases"] -
        0.01 * df["days_since_last_purchase"] +
        1.2 * df["is_promo_active"] -
        0.005 * df["item_price"] ** 1.1
    )
    noise = rng.normal(0, 1.0, size=n)
    df["target"] = np.clip(np.exp(base) + noise, 0, None)  # positive continuous target
    return df

def train_lightgbm_regressor(X_train, y_train, X_val, y_val, params=None):
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": 42
        }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=1000, early_stopping_rounds=40, verbose_eval=100)
    return model

def compute_shap_and_save(model, X_train, X_test, outdir, sample_idx=None):
    """
    Compute SHAP TreeExplainer values, save:
      - summary plot (shap_summary.png)
      - waterfall (local) for sample_idx (if not given chooses a reasonable index)
      - top features csv (top_features.csv)
    Returns the DataFrame of features with mean_abs_shap.
    """
    os.makedirs(outdir, exist_ok=True)
    print("Creating SHAP explainer... (this may take a few seconds)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features) for single-output models

    # handle potential shap output shapes
    if isinstance(shap_values, list):
        # multiclass returned as list -> take class 1 or sum absolute across classes
        shap_arr = np.abs(np.vstack(shap_values)).mean(axis=0)
        shap_values_arr = np.vstack(shap_values)  # fallback structure
        # but for plotting, choose the first class's values where appropriate
        plot_shap_values = shap_values[0] if len(shap_values) > 0 else shap_values[0]
    else:
        shap_values_arr = shap_values
        plot_shap_values = shap_values

    # Global summary plot
    print("Saving SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(plot_shap_values, X_test, show=False)
        plt.tight_layout()
        summary_path = os.path.join(outdir, "shap_summary.png")
        plt.savefig(summary_path, dpi=150)
        plt.close()
        print(f"Saved SHAP summary to: {summary_path}")
    except Exception as e:
        print("Could not save SHAP summary plot:", e)

    # Mean absolute SHAP per feature
    # If shap_values_arr is 2D: (n_samples, n_features)
    if isinstance(shap_values_arr, np.ndarray) and shap_values_arr.ndim == 2:
        mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
    else:
        # fallback
        mean_abs_shap = np.mean(np.abs(plot_shap_values), axis=0)

    feat_imp_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_csv = os.path.join(outdir, "top_features.csv")
    feat_imp_df.to_csv(top_csv, index=False)
    print(f"Saved top features to: {top_csv}")

    # Local waterfall / force plot for a sample index
    if sample_idx is None:
        # choose a sample with large predicted value or middle index
        sample_idx = min(100, max(0, int(len(X_test) // 10)))
    try:
        print(f"Saving SHAP waterfall for sample index {sample_idx} ...")
        sample_path = os.path.join(outdir, f"shap_waterfall_sample{sample_idx}.png")
        plt.figure(figsize=(8, 4))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values_arr[sample_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[sample_idx]
        ), show=False)
        plt.tight_layout()
        plt.savefig(sample_path, dpi=150)
        plt.close()
        print(f"Saved SHAP waterfall to: {sample_path}")
    except Exception as e:
        print("Could not produce SHAP waterfall:", e)

    return feat_imp_df, explainer, shap_values_arr

def generate_template_summary(top_features_df, n_top=5):
    """
    Produce a simple template-based natural language summary from top features.
    """
    top = top_features_df.head(n_top)
    lines = []
    lines.append("Explainability Summary (template-generated)")
    lines.append(f"Generated at: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("Top contributing features (by mean absolute SHAP):")
    for idx, row in top.iterrows():
        feat = row['feature']
        val = float(row['mean_abs_shap'])
        lines.append(f"  {idx+1}. {feat} (mean |SHAP| = {val:.4f})")
    lines.append("")
    lines.append("Short narrative (template):")
    lines.append("  The model's predictions are dominated by the features listed above. Higher mean |SHAP| indicates a stronger average influence on model output.")
    lines.append("  For individual predictions, these features either increase or decrease the predicted value depending on their signs in the SHAP waterfall plots.")
    lines.append("")
    lines.append("Suggested actions:")
    lines.append("  - Inspect top features for potential data quality issues.")
    lines.append("  - Use Partial Dependence Plots for each top feature to understand directionality and thresholds.")
    lines.append("  - For regulated applications, log these explanations per prediction for auditability.")
    return "\n".join(lines)

def call_openai_summary(top_features_df, model_stats):
    """
    Optional: call OpenAI to produce a nicer narrative. Requires openai package and OPENAI_API_KEY env var.
    If not available, this function returns None.
    """
    try:
        import openai
    except Exception as e:
        print("openai package not installed; skipping OpenAI-enhanced summary.")
        return None

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set; skipping OpenAI-enhanced summary.")
        return None

    openai.api_key = key

    # Build a prompt using top features
    top_list = top_features_df.head(8).to_dict(orient="records")
    prompt = (
        "You are an AI assistant that writes clear, concise natural-language explanations of model feature importances."
        "Given the following top features and their mean absolute SHAP values, craft a 5-7 sentence executive summary describing which features drive model predictions,"
        "what that implies for actions, and a short " + "`next steps`" + " checklist. Also include a one-line caution about interpretability limits.\n\n"
        f"Model stats: RMSE={model_stats.get('rmse', 'NA'):.4f}, R2={model_stats.get('r2', 'NA'):.4f}\n\n"
        "Top features:\n"
    )
    for rec in top_list:
        prompt += f"- {rec['feature']}: mean_abs_shap={rec['mean_abs_shap']:.6f}\n"

    # system / message style
    messages = [
        {"role": "system", "content": "You are a succinct AI summarizer for model explainability."},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, max_tokens=400, temperature=0.2)
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        print("OpenAI call failed:", e)
        return None

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def main(outdir="explainability_bundle"):
    print("1) Simulating dataset...")
    df = simulate_regression_data(n=20000, seed=42)
    features = ["age", "income", "num_prev_purchases", "days_since_last_purchase", "is_promo_active", "item_price"]
    target = "target"

    X = df[features]
    y = df[target]

    print("2) Splitting data (train/val/test)...")
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("3) Training LightGBM regressor...")
    model = train_lightgbm_regressor(X_train, y_train, X_val, y_val)
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    print("4) Evaluate on test set...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"  RMSE: {rmse:.4f}  R2: {r2:.4f}")
    sample_cmp = pd.DataFrame({"actual": y_test.values[:20], "pred": preds[:20]})
    sample_cmp.to_csv(os.path.join(outdir, "sample_predictions.csv"), index=False)

    model_stats = {"rmse": rmse, "r2": r2}

    print("5) Computing SHAP and saving artifacts...")
    feat_imp_df, explainer, shap_vals = compute_shap_and_save(model, X_train, X_test, outdir, sample_idx=10)

    print("6) Generating template-based natural-language summary...")
    template_summary = generate_template_summary(feat_imp_df, n_top=6)
    summary_path = os.path.join(outdir, "nl_summary.txt")
    save_text(summary_path, template_summary)
    print(f"Saved template summary to {summary_path}")

    # Optionally call OpenAI for a richer narrative if available
    if USE_OPENAI_IF_AVAILABLE and OPENAI_API_KEY:
        print("7) OpenAI key found: attempting to create enhanced natural language summary via OpenAI...")
        openai_text = call_openai_summary(feat_imp_df, model_stats)
        if openai_text:
            openai_path = os.path.join(outdir, "nl_summary_openai.txt")
            save_text(openai_path, openai_text)
            print(f"Saved OpenAI-enhanced summary to {openai_path}")
        else:
            print("OpenAI summary not produced (call failed or not available).")

    print("All done. Check the output folder:", outdir)
    print("Files created:")
    for f in sorted(os.listdir(outdir)):
        print("  -", f)

if __name__ == "__main__":
    main()
