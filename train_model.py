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