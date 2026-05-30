# PCOS-AI-Risk-Prediction

An end-to-end machine learning system to predict **Polycystic Ovary Syndrome (PCOS)** risk using real patient data from 10 hospitals in Kerala, India. Built with Python, Scikit-learn, XGBoost, SHAP, and Streamlit.

> **MSc Advanced Data Science & AI** — University of Liverpool | Chaitra Shanthamallaiah | May 2026

-----

## Project Overview

PCOS affects 1 in 10 women of reproductive age and is often under-diagnosed. This project builds a clinical decision-support tool that predicts PCOS risk from patient symptoms and hormone levels, with explainable AI (SHAP) to show *why* each prediction was made.

**Best model: Logistic Regression — 90.5% cross-validated accuracy**

-----

## Features

- Compares 4 ML models: Random Forest, Logistic Regression, SVM, XGBoost
- Feature selection: 15 clinically relevant features from 45
- StandardScaler preprocessing (boosted SVM by +20%)
- 5-fold cross-validation for reliable accuracy estimates
- SHAP values for explainable AI — shows *why* the model predicts PCOS
- Interactive Streamlit web app for real-time risk prediction

-----

##  Model Performance

|Model                  |Test Accuracy|CV Accuracy |
|-----------------------|-------------|------------|
|Random Forest          |88.25%       |88.25%      |
|**Logistic Regression**|**89.81%**   |**90.50% 🏆**|
|SVM                    |88.89%       |88.62%      |
|XGBoost                |87.96%       |85.46%      |


> SVM without scaling: 68.52% → with scaling: 88.89% (+20.37%)

-----

## Key PCOS Predictors Found

1. Follicle No. (Right ovary) 🥇
1. Follicle No. (Left ovary)
1. Skin darkening
1. Hair growth
1. Weight gain
1. AMH hormone
1. Cycle regularity
1. BMI
1. LH hormone
1. Cycle length

-----

## Project Structure

```
pcos-ai-risk-prediction/
│
├── explore_data.py          # Data loading and exploration (541 patients, 45 features)
├── model.py                 # Final model with StandardScaler + 5-fold CV
├── visualise.py             # Distribution and feature importance charts
├── compare_models.py        # Side-by-side comparison of all 4 models
├── app.py                   # Streamlit web application
│
├── charts/
│   ├── chart1_distribution.png     # PCOS patient distribution
│   ├── chart2_features.png         # Top 10 feature importance
│   └── chart3_model_comparison.png # Model accuracy comparison
│
├── requirements.txt
└── README.md
```

-----

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Chaitra-ML/PCOS-AI-Risk-Prediction
cd pcos-ai-risk-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the PCOS dataset from [Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) and place it in the project root.

### 4. Run the scripts in order

```bash
python explore_data.py       # Explore the dataset
python visualise.py          # Generate charts
python compare_models.py     # Compare all models
python model.py              # Train final model
```

### 5. Launch the web app

```bash
streamlit run app.py
```

-----

##  Explainable AI — SHAP

SHAP (SHapley Additive exPlanations) values are used to explain individual predictions. This is critical for clinical trust — a doctor can see exactly which symptoms drove a high-risk prediction.

```python
import shap
explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test, feature_names=features)
```

-----

##  Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
streamlit
```

Install all at once:

```bash
pip install -r requirements.txt
```

-----

##  Key Lessons Learned

|Finding                             |Lesson                                            |
|------------------------------------|--------------------------------------------------|
|Feature selection (42 → 15 features)|Better data beats better algorithms               |
|StandardScaler on SVM               |Preprocessing matters as much as model choice     |
|5-fold cross-validation             |Simple train/test split can be misleading         |
|Logistic Regression won             |Simple models can beat complex ones with good data|

-----

## Future Work

### Completed
- [x] Deploy on **Streamlit Cloud** for public access
- [x] Add **SHAP waterfall plots** per patient in the web app
- [x] Explore **Federated Learning** across multiple hospital datasets (privacy-preserving)

### Clinical Improvements
- [ ] Multi-class severity prediction (low / moderate / high risk) instead of binary yes/no
- [ ] Age-group analysis — does model accuracy differ for teens vs adults?
- [ ] Incorporate **ultrasound image data (CNN)** alongside tabular data — multimodal AI

### Technical Enhancements
- [ ] Hyperparameter tuning with **GridSearchCV / Optuna** for all 4 models
- [ ] **Ensemble stacking** — combine all 4 models for potentially higher accuracy
- [ ] Add **LIME explanations** alongside SHAP for comparison
- [ ] **Bias detection** — test model fairness across demographic groups

###  Deployment & Integration
- [ ] Build a **REST API using FastAPI** for integration into hospital systems
- [ ] Add **user authentication** to the Streamlit app for clinical use
- [ ] Add **confidence intervals** to risk predictions



-----

##  Dataset

- **Source:** [Kaggle — PCOS Dataset](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)
- **Size:** 541 patients, 45 features
- **Origin:** 10 hospitals across Kerala, India
- **Distribution:** 364 without PCOS, 177 with PCOS

-----

##  Author

**Chaitra Shanthamallaiah**
MSc Advanced Data Science & AI, University of Liverpool
May 2026

-----

##  License

This project is licensed under the MIT License — see <LICENSE> for details.

-----

##  Acknowledgements

- Dataset: Prasoon Kottarathil via Kaggle
- SHAP library: Lundberg & Lee (2017)
- University of Liverpool — MSc ADSA programme