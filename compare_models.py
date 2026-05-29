import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_excel("archive/PCOS_data_without_infertility.xlsx", sheet_name=1)
df = df.drop(columns=['Sl. No', 'Patient File No.', 'Unnamed: 44'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

X = df.drop(columns=['PCOS (Y/N)'])
y = df['PCOS (Y/N)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# Train and compare
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = round(acc * 100, 2)
    print(f"{name}: {acc*100:.2f}%")

# Plot comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['green','blue','orange','red'])
plt.title('Model Comparison - Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(70, 100)
plt.tight_layout()
plt.savefig('chart3_model_comparison.png')
print("\nChart saved!")