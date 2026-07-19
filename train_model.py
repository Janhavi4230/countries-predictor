import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("Countries.csv")

print(df.head())

print(df.info())
print(df.isnull().sum())

df = df.dropna()

print(df.shape)
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    # Target and Features
X = df.drop("life_expectancy", axis=1)
y = df["life_expectancy"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
# Save Model
joblib.dump(model, "model.pkl")

# Save Label Encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Save Feature Names
joblib.dump(list(X.columns), "feature_names.pkl")

print("Model Saved Successfully")