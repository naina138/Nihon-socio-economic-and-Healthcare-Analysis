import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from joblib import load

svm_model = load("svm_model.pkl")
train_df = pd.read_csv("train.csv")
feature_names = train_df.drop(columns=['Cluster']).columns
coefs = svm_model.coef_

for i, class_label in enumerate(svm_model.classes_):
    plt.figure(figsize=(10, 6))
    coef_series = pd.Series(coefs[i], index=feature_names)
    coef_series.sort_values().plot(kind='barh', color='skyblue', title=f"Feature Importance for Class {class_label}")
    plt.xlabel("Coefficient Value")
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    filename = f"feature_importance_class_{class_label}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    
    plt.show()
    plt.close()

