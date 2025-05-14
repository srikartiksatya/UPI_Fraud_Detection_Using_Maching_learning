import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv('upi_fraud_dataset.csv')

# Encode categorical variables if needed
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature selection (assume all columns except 'fraud' are features)
X = data.drop(columns=['fraud'], errors='ignore')
y = data['fraud'] if 'fraud' in data.columns else None

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define and train HMM model
n_components = 2  # Assume 2 states: fraud and non-fraud
model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
model.fit(X_scaled)

# Function to predict fraud likelihood for a new transaction
def predict_fraud(new_transaction):
    new_transaction = np.array(new_transaction).reshape(1, -1)
    new_transaction_scaled = scaler.transform(new_transaction)
    log_prob, _ = model.decode(new_transaction_scaled)
    return log_prob

# Example input transaction (replace with actual data)
example_transaction = X.iloc[5].values  # Taking first transaction as an example
prediction = predict_fraud(example_transaction)

print("Fraud likelihood score:", prediction)
