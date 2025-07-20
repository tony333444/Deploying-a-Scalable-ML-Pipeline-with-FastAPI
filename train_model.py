import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "..", "data", "census.csv")
data = pd.read_csv(data_path)

# Split the provided data to have a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Ensure model directory exists
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

# Save the model, encoder, and label binarizer
model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)
lb_path = os.path.join(model_dir, "lb.pkl")
save_model(lb, lb_path)

# Load the model
model = load_model(model_path)

# Run inference on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
            )
            f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n\n")
