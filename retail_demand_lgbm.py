"""
retail_demand_lgbm.py
End-to-end LightGBM demand forecasting demo for retail.
- Simulates inventory & sales for multiple stores/products
- Feature engineering (lags, moving averages, promo flags, weather, holidays)
- Trains LightGBM regression model (predict sales_units)
- Evaluates model and saves metrics
- Produces SHAP, PDP, and LIME explainability artifacts
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import shap
from sklearn.inspection import PartialDependenceDisplay
from lime import lime_tabular

RND = 42
np.random.seed(RND)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def simulate_retail_data(n_stores=30, n_products=50, days=365, seed=RND):
    """
    Simulate daily sales data for multiple stores and products.
    Returns a DataFrame with columns:
        date, store_id, product_id, price, promo_flag, temperature, is_holiday, sales_units
    """
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    rows = []
    base_demand_by_product = np.random.uniform(5, 200, size=n_products)  # base daily demand
    price_sensitivity = np.random.uniform(-0.5, -0.05, size=n_products)  # elasticity

    for store in range(n_stores):
        store_scale = np.random.uniform(0.6, 1.4)  # store popularity
        region_temp_bias = np.random.normal(0, 3)  # local climate shift
        for product in range(n_products):
            base = base_demand_by_product[product] * store_scale
            elasticity = price_sensitivity[product]
            for d in dates:
                day_of_week = d.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                # Promotions occur randomly, more on weekends
                promo_flag = np.random.binomial(1, 0.08 + 0.05 * is_weekend)
                # Price varies with promo
                base_price = np.round(np.random.uniform(50, 200), 2)
                if promo_flag:
                    price = np.round(base_price * np.random.uniform(0.6, 0.9), 2)
                else:
                    price = base_price
                # Weather (temperature) seasonal: sin wave + noise
                day_index = (d - dates[0]).days
                seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_index / 365) + region_temp_bias
                temperature = seasonal_temp + np.random.normal(0, 3)
                # Holiday flag: assume some fixed holidays
                is_holiday = 1 if (d.month == 12 and d.day in [24,25,31]) else 0
                # Promotion lift & holiday lift & weekend effect
                promo_lift = np.random.uniform(1.2, 2.5) if promo_flag else 1.0
                holiday_lift = np.random.uniform(1.1, 1.6) if is_holiday else 1.0
                weekend_lift = 1.15 if is_weekend else 1.0
                # Temperature effect: for cold-weather products, higher temp lowers demand (we'll emulate mixed products)
                temp_effect = 1.0 + 0.01 * (temperature - 20)  # simple linear effect
                # Final expected demand before noise:
                expected = max(0.1, base * promo_lift * holiday_lift * weekend_lift * temp_effect *
                               (1.0 + elasticity * (price - base_price)/base_price))
                # Poisson noise for counts:
                sales_units = np.random.poisson(lam=max(0.001, expected))
                rows.append({
                    "date": d,
                    "store_id": f"store_{store:03d}",
                    "product_id": f"prod_{product:04d}",
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "price": price,
                    "base_price": base_price,
                    "promo_flag": promo_flag,
                    "temperature": round(temperature, 2),
                    "is_holiday": is_holiday,
                    "sales_units": sales_units
                })
    df = pd.DataFrame(rows)
    return df

def add_lag_features(df, lags=[1,7,14,28]):
    # sort and create lags per (store,product)
    df = df.sort_values(["store_id","product_id","date"]).reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_id","product_id"])["sales_units"].shift(lag)
    # rolling features
    df["rolling_7_mean"] = df.groupby(["store_id","product_id"])["sales_units"].shift(1).rolling(window=7, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df["rolling_30_mean"] = df.groupby(["store_id","product_id"])["sales_units"].shift(1).rolling(window=30, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    # days since last non-zero sale (to capture inactivity)
    df["prev_nonzero_days"] = df.groupby(["store_id","product_id"])["sales_units"].apply(lambda x: (x == 0).cumsum() - (x == 0).cumsum().where(x != 0).ffill().fillna(0).astype(int))
    return df

def encode_and_clean(df):
    # One-hot small categorical features or label-encode store/product
    df["store_idx"] = df["store_id"].astype('category').cat.codes
    df["product_idx"] = df["product_id"].astype('category').cat.codes
    # day_of_week is already numeric
    # Replace NaNs in lags with 0
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df[lag_cols] = df[lag_cols].fillna(0)
    df["rolling_7_mean"] = df["rolling_7_mean"].fillna(0)
    df["rolling_30_mean"] = df["rolling_30_mean"].fillna(0)
    df["prev_nonzero_days"] = df["prev_nonzero_days"].fillna(0)
    return df

def prepare_features(df):
    # Create price discount percent
    df["price_disc_pct"] = (df["base_price"] - df["price"]) / df["base_price"]
    # Interaction features
    df["promo_weekend"] = df["promo_flag"] * df["is_weekend"]
    df["temp_promo_inter"] = df["temperature"] * df["promo_flag"]
    # target
    df["target"] = df["sales_units"]
    # Drop columns not used
    drop_cols = ["store_id","product_id","sales_units","date"]
    X = df.drop(columns=drop_cols + ["target"], errors='ignore')
    y = df["target"]
    return X, y

def train_lightgbm_regression(X_train, y_train, X_val, y_val):
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
        "seed": RND
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=1000, early_stopping_rounds=40, verbose_eval=50)
    return model

def evaluate_and_save(model, X_test, y_test):
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n")
    print("EVAL -> RMSE: %.4f  MAE: %.4f  R2: %.4f" % (rmse, mae, r2))
    # sample compare
    sample_comp = pd.DataFrame({"actual": y_test.values, "pred": preds})
    sample_comp.head(10).to_csv(os.path.join(OUT_DIR, "sample_predictions.csv"), index=False)
    joblib.dump(model, os.path.join(OUT_DIR, "lgbm_retail_model.pkl"))
    print("Saved model and sample predictions.")
    return preds

def explainability_artifacts(model, X_train, X_test, sample_idx=100):
    # SHAP global
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    sfile = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(sfile, dpi=150)
    plt.close()
    print("Saved SHAP summary:", sfile)

    # SHAP waterfall for a local sample
    try:
        sw = os.path.join(OUT_DIR, f"shap_waterfall_sample{sample_idx}.png")
        plt.figure(figsize=(8,4))
        shap.plots.waterfall(shap.Explanation(values=shap_values[sample_idx],
                                             base_values=explainer.expected_value,
                                             data=X_test.iloc[sample_idx]), show=False)
        plt.tight_layout()
        plt.savefig(sw, dpi=150)
        plt.close()
        print("Saved SHAP waterfall:", sw)
    except Exception as e:
        print("Could not generate SHAP waterfall:", e)

    # PDP for price_disc_pct and temperature
    for feat in ["price_disc_pct","temperature"]:
        if feat in X_train.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            PartialDependenceDisplay.from_estimator(model, X_train, [feat], ax=ax)
            pdp_file = os.path.join(OUT_DIR, f"pdp_{feat}.png")
            plt.tight_layout()
            plt.savefig(pdp_file, dpi=150)
            plt.close()
            print("Saved PDP:", pdp_file)

    # LIME explanation for the same local sample
    try:
        feature_names = X_train.columns.tolist()
        expl = lime_tabular.LimeTabularExplainer(training_data=X_train.values,
                                                 feature_names=feature_names,
                                                 mode='regression',
                                                 discretize_continuous=True,
                                                 random_state=RND)
        exp = expl.explain_instance(X_test.iloc[sample_idx].values, model.predict, num_features=8)
        html_file = os.path.join(OUT_DIR, "lime_explanation.html")
        exp.save_to_file(html_file)
        print("Saved LIME explanation:", html_file)
    except Exception as e:
        print("Could not generate LIME explanation:", e)

def main():
    print("Simulating retail dataset...")
    df = simulate_retail_data(n_stores=15, n_products=30, days=400)
    print("Rows simulated:", len(df))

    print("Adding lag features...")
    df = add_lag_features(df, lags=[1,7,14,28])

    print("Encoding & cleaning...")
    df = encode_and_clean(df)

    print("Preparing features...")
    X, y = prepare_features(df)
    # Drop rows where lag_1 is NaN (early dates) to ensure good training
    mask = ~X["lag_1"].isna()
    X = X[mask]
    y = y[mask]
    print("Prepared feature matrix:", X.shape)

    # train/val/test split by time: keep temporal order
    # We'll split by the last 20% as test, preceding 10% as val
    n = X.shape[0]
    test_size = int(0.2 * n)
    val_size = int(0.1 * n)
    X_train = X.iloc[: n - test_size - val_size]
    y_train = y.iloc[: n - test_size - val_size]
    X_val = X.iloc[n - test_size - val_size: n - test_size]
    y_val = y.iloc[n - test_size - val_size: n - test_size]
    X_test = X.iloc[n - test_size:]
    y_test = y.iloc[n - test_size:]
    print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    print("Training LightGBM regression...")
    model = train_lightgbm_regression(X_train, y_train, X_val, y_val)

    print("Evaluating and saving model...")
    preds = evaluate_and_save(model, X_test, y_test)

    print("Generating explainability artifacts (SHAP, PDP, LIME)...")
    explainability_artifacts(model, X_train, X_test, sample_idx=150)

    print("All done. Check the outputs/ folder for metrics, model, and visualizations.")

if __name__ == "__main__":
    main()
