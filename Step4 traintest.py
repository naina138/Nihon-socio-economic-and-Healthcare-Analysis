import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

df = pd.read_csv("clustered_data.csv")

non_numeric_cols = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_cols)

X = df.drop(columns=['Cluster'])
y = df['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_df = X_train.copy()
train_df['Cluster'] = y_train
train_df.to_csv("train.csv", index=False)

test_df = X_test.copy()
test_df['Cluster'] = y_test
test_df.to_csv("test.csv", index=False)

print("Cleaned and split saved: 'train.csv' and 'test.csv'")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved")
