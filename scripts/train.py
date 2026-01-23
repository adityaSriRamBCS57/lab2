# import os
# import json
# import joblib
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, r2_score

# # Ensure output directory exists
# os.makedirs("output", exist_ok=True)

# # Load dataset
# df = pd.read_csv("data/winequality-red.csv", sep=";")

# X = df.drop("quality", axis=1)
# y = df["quality"]


# X_scaled = X.values

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.4, random_state=42
# )

# # Model (RIDGE)
# model = Ridge(alpha=1.0)
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)

# # Metrics
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"MSE: {mse}")
# print(f"R2: {r2}")

# # Save outputs
# joblib.dump(model, "output/model.pkl")

# results = {
#     "MSE": mse,
#     "R2": r2
# }

# with open("output/results.json", "w") as f:
#     json.dump(results, f, indent=4)

# print("Saved files:", os.listdir("output"))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")
data = pd.concat([red, white])

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics
metrics = {
    "mse": mse,
    "r2": r2
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Training completed")
print(metrics)

