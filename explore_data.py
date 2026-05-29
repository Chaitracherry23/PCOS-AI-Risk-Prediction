import pandas as pd

# Load main file only
df = pd.read_excel("archive/PCOS_data_without_infertility.xlsx", sheet_name=1)

print("Shape:", df.shape)
print("\nPCOS distribution:")
print(df['PCOS (Y/N)'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())