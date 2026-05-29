import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# App UI
st.title("🌸 PCOS Risk Prediction App")
st.write("Enter patient details below:")

age = st.slider("Age (years)", 18, 50, 25)
weight = st.slider("Weight (Kg)", 30, 120, 60)
bmi = st.slider("BMI", 15.0, 45.0, 22.0)
cycle = st.selectbox("Cycle (2=Regular, 4=Irregular)", [2, 4])
follicle_l = st.slider("Follicle No. (L)", 0, 20, 5)
follicle_r = st.slider("Follicle No. (R)", 0, 20, 5)
weight_gain = st.selectbox("Weight gain?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
hair_growth = st.selectbox("Excess hair growth?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
skin_dark = st.selectbox("Skin darkening?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
hair_loss = st.selectbox("Hair loss?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
pimples = st.selectbox("Pimples?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
fast_food = st.selectbox("Fast food regularly?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
amh = st.slider("AMH (ng/mL)", 0.0, 10.0, 2.0)
fsh = st.slider("FSH (mIU/mL)", 0.0, 20.0, 5.0)
lh = st.slider("LH (mIU/mL)", 0.0, 20.0, 5.0)

if st.button("Predict PCOS Risk"):
    input_df = pd.DataFrame([[age, weight, bmi, cycle,
                               follicle_l, follicle_r,
                               weight_gain, hair_growth,
                               skin_dark, hair_loss,
                               pimples, fast_food,
                               amh, fsh, lh]], columns=features)
    input_scaled = scaler.transform(input_df)
    result = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if result == 1:
        st.error(f"⚠️ High PCOS Risk: {prob*100:.1f}% probability")
    else:
        st.success(f"✅ Low PCOS Risk: {prob*100:.1f}% probability")