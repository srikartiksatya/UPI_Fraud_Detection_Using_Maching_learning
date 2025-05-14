import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('upi_fraud_dataset.csv')

# Handle missing values
df.fillna(df.median(), inplace=True)

# Splitting features and target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using RFE
selector = RFE(LogisticRegression(max_iter=200), n_features_to_select=8)
selector.fit(X_train_scaled, y_train)
X_train_selected = X_train_scaled[:, selector.support_]
X_test_selected = X_test_scaled[:, selector.support_]

# Train models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

accuracy_scores = {}
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    accuracy_scores[name] = accuracy_score(y_test, y_pred)

# Identify and save best model
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "selector.pkl")

# Accuracy comparison plot
plt.figure(figsize=(8, 5))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue', 'green', 'red'])
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Comparison of Model Accuracies (Optimized Features)")
plt.ylim(0, 1)
plt.show()

# Load best model and predict new input
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# New input for prediction
new_input = np.array([[12, 1, 1, 2, 2022, 4, 9132000002, 30, 281.06, 27, 28611]])
new_input_scaled = scaler.transform(new_input)
new_input_selected = new_input_scaled[:, selector.support_]

prediction = best_model.predict(new_input_selected)
print("Prediction result:", prediction[0])
