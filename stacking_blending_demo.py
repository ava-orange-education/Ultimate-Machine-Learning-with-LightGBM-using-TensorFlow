"""
stacking_blending_demo.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

CHATGPT_PROMPT = """
Generate a Python example demonstrating Stacking and Blending
using Random Forest, Gradient Boosting, and Logistic Regression.
"""

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

estimators = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stack.fit(X_train, y_train)
print("Stacking Accuracy:",
      accuracy_score(y_test, stack.predict(X_test)))

X_base, X_hold, y_base, y_hold = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf.fit(X_base, y_base)
gb.fit(X_base, y_base)

blend_train = np.column_stack([
    rf.predict_proba(X_hold)[:,1],
    gb.predict_proba(X_hold)[:,1]
])

meta = LogisticRegression(max_iter=1000)
meta.fit(blend_train, y_hold)

blend_test = np.column_stack([
    rf.predict_proba(X_test)[:,1],
    gb.predict_proba(X_test)[:,1]
])

print("Blending Accuracy:",
      accuracy_score(y_test, meta.predict(blend_test)))
