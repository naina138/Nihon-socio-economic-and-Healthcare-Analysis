import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

test_df = pd.read_csv("test.csv")
X_test = test_df.drop(columns=['Cluster'])
y_test = test_df['Cluster']
scaler = joblib.load("scaler.pkl")
svm_model = joblib.load("svm_model.pkl")
X_test_scaled = scaler.transform(X_test)
y_pred = svm_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - SVM Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
