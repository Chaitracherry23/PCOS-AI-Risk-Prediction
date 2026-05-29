import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load and clean data
df = pd.read_excel("archive/PCOS_data_without_infertility.xlsx", sheet_name=1)
df = df.drop(columns=['Sl. No', 'Patient File No.', 'Unnamed: 44'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# SHAP explanation
explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer.shap_values(X_scaled)

# Plot 1 - Summary plot
plt.figure()
shap.summary_plot(shap_values, X, feature_names=features, show=False)
plt.tight_layout()
plt.savefig('chart4_shap_summary.png', bbox_inches='tight')
print("SHAP chart saved!")