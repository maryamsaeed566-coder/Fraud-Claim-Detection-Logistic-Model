import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

st.title("🚨 Insurance Fraud Detection App")
st.write("Logistic Regression based Fraud Prediction")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_excel("fraud_insurance_claims.xls")
    return df


df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------- PREPROCESSING --------------------
le = LabelEncoder()
df['fraud_reported'] = le.fit_transform(
    df['fraud_reported'].astype(str).str.strip()
)

df.fillna(method='ffill', inplace=True)

drop_cols = [
    'policy_number',
    'policy_bind_date',
    'incident_date',
    'insured_zip',
    'incident_location'
]

df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

cat_cols = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# -------------------- MODEL --------------------
X = df_encoded.drop('fraud_reported', axis=1)
y = df_encoded['fraud_reported']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------- EVALUATION --------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Model Performance")
st.text(classification_report(y_test, y_pred))
st.metric("ROC-AUC", round(roc_auc_score(y_test, y_prob), 3))

# -------------------- USER INPUT --------------------
st.subheader("Predict Fraud for New Claim")

inputs = {}
for col in X.columns[:8]:  # keep simple
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"🚨 Fraud Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Claim (Fraud Probability: {prob:.2f})")
