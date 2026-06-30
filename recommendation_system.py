
"""
Recommendation System – Executable Python Script
------------------------------------------------
This script trains a hybrid recommendation model using implicit feedback.
It uses user embeddings, item embeddings, and item category context.

Requirements:
- Python 3.8+
- tensorflow
- pandas
- numpy

Execution:
python recommendation_system.py
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------
# Load Dataset
# -----------------------------
# Expected CSV columns:
# user_id, item_id, category_id, interaction_strength

data = pd.read_csv("interactions.csv")

# Shuffle data to avoid ordering bias
data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Cardinalities
num_users = data["user_id"].nunique()
num_items = data["item_id"].nunique()
num_categories = data["category_id"].nunique()

# -----------------------------
# Train / Validation Split
# -----------------------------
split_idx = int(0.8 * len(data))
train_data = data.iloc[:split_idx]
val_data = data.iloc[split_idx:]

# -----------------------------
# Model Parameters
# -----------------------------
EMBEDDING_DIM = 64

# -----------------------------
# User Embedding
# -----------------------------
user_input = layers.Input(shape=(1,), name="user_id")
user_embedding = layers.Embedding(
    input_dim=num_users + 1,
    output_dim=EMBEDDING_DIM,
    name="user_embedding"
)(user_input)
user_vector = layers.Flatten()(user_embedding)

# -----------------------------
# Item Embedding
# -----------------------------
item_input = layers.Input(shape=(1,), name="item_id")
item_embedding = layers.Embedding(
    input_dim=num_items + 1,
    output_dim=EMBEDDING_DIM,
    name="item_embedding"
)(item_input)
item_vector = layers.Flatten()(item_embedding)

# -----------------------------
# Category Embedding
# -----------------------------
category_input = layers.Input(shape=(1,), name="category_id")
category_embedding = layers.Embedding(
    input_dim=num_categories + 1,
    output_dim=32,
    name="category_embedding"
)(category_input)
category_vector = layers.Flatten()(category_embedding)

# -----------------------------
# Combine Item Context
# -----------------------------
item_context = layers.Concatenate()([item_vector, category_vector])
item_context = layers.Dense(128, activation="relu")(item_context)

# -----------------------------
# User-Item Interaction Network
# -----------------------------
interaction = layers.Concatenate()([user_vector, item_context])
interaction = layers.Dense(128, activation="relu")(interaction)
interaction = layers.Dense(64, activation="relu")(interaction)

# -----------------------------
# Output Layer
# -----------------------------
output = layers.Dense(1, activation="linear", name="score")(interaction)

# -----------------------------
# Build and Compile Model
# -----------------------------
model = Model(
    inputs=[user_input, item_input, category_input],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    x=[
        train_data["user_id"],
        train_data["item_id"],
        train_data["category_id"]
    ],
    y=train_data["interaction_strength"],
    validation_data=(
        [
            val_data["user_id"],
            val_data["item_id"],
            val_data["category_id"]
        ],
        val_data["interaction_strength"]
    ),
    epochs=10,
    batch_size=256
)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_items(user_id, top_k=5):
    """
    Recommend top K items for a given user
    """
    candidate_items = np.arange(1, num_items + 1)
    user_ids = np.full(len(candidate_items), user_id)

    category_lookup = (
        data.groupby("item_id")["category_id"]
        .first()
        .to_dict()
    )

    category_ids = [category_lookup[i] for i in candidate_items]

    scores = model.predict(
        [user_ids, candidate_items, category_ids],
        verbose=0
    ).flatten()

    ranked_items = sorted(
        zip(candidate_items, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_items[:top_k]

# -----------------------------
# Example Inference
# -----------------------------
if __name__ == "__main__":
    user_id = 10
    recommendations = recommend_items(user_id)

    print(f"Top recommendations for user {user_id}:")
    for item, score in recommendations:
        print(f"Item {item} -> Score {round(score, 2)}")
