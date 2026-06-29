"""
hyperopt_hyperparameter_tuning.py

Demonstrates how ChatGPT can help generate and automate
hyperparameter tuning experiments using Hyperopt.
"""

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

CHATGPT_PROMPT = """
Generate a Python script using Hyperopt to optimize a RandomForestClassifier.
Requirements:
- Use the Breast Cancer dataset from scikit-learn.
- Optimize n_estimators, max_depth, min_samples_split, and min_samples_leaf.
- Use 5-fold cross-validation.
- Maximize classification accuracy.
- Save the best model using joblib.
"""

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

space = {
    "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
    "max_depth": hp.quniform("max_depth", 3, 20, 1),
    "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
}

def objective(params):
    params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    ).mean()

    return {"loss": -score, "status": STATUS_OK}

trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials,
)

best_params = {
    "n_estimators": int(best["n_estimators"]),
    "max_depth": int(best["max_depth"]),
    "min_samples_split": int(best["min_samples_split"]),
    "min_samples_leaf": int(best["min_samples_leaf"]),
    "random_state": 42,
    "n_jobs": -1,
}

print("Best Hyperparameters")
print(best_params)

model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, pred):.4f}")

joblib.dump(model, "best_random_forest_model.pkl")
print("Model saved as best_random_forest_model.pkl")

print("\\nSuggested follow-up ChatGPT prompts:")
print("- Add early stopping where supported.")
print("- Replace Random Forest with LightGBM.")
print("- Log all evaluations to CSV.")
print("- Compare Hyperopt with Optuna.")
