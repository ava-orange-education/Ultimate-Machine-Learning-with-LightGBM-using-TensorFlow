"""
optuna_hyperparameter_tuning.py

Demonstrates how ChatGPT can help generate and automate
hyperparameter tuning experiments using Optuna.
"""

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

CHATGPT_PROMPT = """
Generate a Python script using Optuna to optimize a RandomForestClassifier.
Requirements:
1. Use sklearn Breast Cancer dataset.
2. Optimize n_estimators, max_depth, min_samples_split, min_samples_leaf.
3. Use 5-fold cross-validation.
4. Maximize accuracy.
5. Save the best model using joblib.
"""

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    ).mean()

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\\nBest Hyperparameters")
print(study.best_params)
print(f"Best CV Accuracy: {study.best_value:.4f}")

best_model = RandomForestClassifier(
    **study.best_params,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

pred = best_model.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, pred):.4f}")

joblib.dump(best_model, "best_random_forest_model.pkl")
print("Model saved as best_random_forest_model.pkl")

print("\\nSuggested follow-up ChatGPT prompts:")
print("- Add Optuna pruning.")
print("- Replace RandomForestClassifier with LightGBM.")
print("- Enable parallel optimization.")
print("- Log all trials to CSV.")
