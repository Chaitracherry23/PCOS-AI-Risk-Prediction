import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and clean data
df = pd.read_excel("archive/PCOS_data_without_infertility.xlsx", sheet_name=1)
df = df.drop(columns=['Sl. No', 'Patient File No.', 'Unnamed: 44'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Chart 1 - PCOS distribution
plt.figure(figsize=(6,4))
df['PCOS (Y/N)'].value_counts().plot(kind='bar', color=['skyblue','salmon'])
plt.title('PCOS Distribution')
plt.xticks([0,1], ['No PCOS', 'PCOS'], rotation=0)
plt.savefig('chart1_distribution.png')
print("Chart 1 saved!")

# Chart 2 - Feature importance
X = df.drop(columns=['PCOS (Y/N)'])
y = df['PCOS (Y/N)']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns)
top10 = importance.nlargest(10)

plt.figure(figsize=(8,6))
top10.plot(kind='barh', color='mediumseagreen')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('chart2_features.png')
print("Chart 2 saved!")