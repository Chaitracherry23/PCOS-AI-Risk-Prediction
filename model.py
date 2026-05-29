import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load and clean data
df = pd.read_excel("archive/PCOS_data_without_infertility.xlsx", sheet_name=1)
df = df.drop(columns=['Sl. No', 'Patient File No.', 'Unnamed: 44'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Use more symptoms this time
features = [
    ' Age (yrs)', 'Weight (Kg)', 'BMI', 'Cycle(R/I)',
    'Follicle No. (L)', 'Follicle No. (R)',
    'Weight gain(Y/N)', 'hair growth(Y/N)',
    'Skin darkening (Y/N)', 'Hair loss(Y/N)',
    'Pimples(Y/N)', 'Fast food (Y/N)',
    'AMH(ng/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)'
]

X = df[features]
y = df['PCOS (Y/N)']

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

print("=== Model Comparison with StandardScaler ===\n")
for name, model in models.items():
    # Cross validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}:")
    print(f"  Test Accuracy:  {acc*100:.2f}%")
    print(f"  CV Accuracy:    {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n")