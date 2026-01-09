import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Preprocessing (SCALER)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model (RIDGE)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

# Save outputs
joblib.dump(model, "output/model.pkl")

results = {
    "MSE": mse,
    "R2": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Saved files:", os.listdir("output"))
