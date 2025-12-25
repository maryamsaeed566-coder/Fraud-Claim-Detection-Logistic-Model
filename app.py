# ==================== IMPORTS ====================
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")
st.title("🚨 Insurance Fraud Detection App")

# ==================== READ CSV (WORKS LOCALLY + GITHUB) ====================
try:
    df = pd.read_csv("fraud_insurance_claims.csv")
    st.success("✅ CSV loaded successfully")
except Exception as e:
    st.error("❌ fraud_insurance_claims.csv not found")
    st.stop()

# ==================== CLEAN COLUMNS ====================
df.columns = df.columns.str.strip().str.lower()

# ==================== DATA PREVIEW ====================
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ==================== TARGET COLUMN ====================
target_col = st.selectbox("🎯 Select target (fraud) column", df.columns)

if not target_col:
    st.stop()

# ==================== PREPROCESSING ====================
df = df.copy()
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col].astype(str))

df.fillna(method="ffill", inplace=True)

drop_cols = [
    "policy_number",
    "policy_bind_date",
    "incident_date",
    "insured_zip",
    "incident_location"
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

cat_cols = df.select_dtypes(include="object").columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ==================== MODEL DATA ====================
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

stratify = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=stratify
)

# ==================== SCALING ====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================== MODEL ====================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==================== EVALUATION ====================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("📊 Model Performance")
st.text(classification_report(y_test, y_pred))
st.metric("ROC-AUC Score", round(roc_auc_score(y_test, y_prob), 3))

# ==================== PREDICTION ====================
st.subheader("🔍 Predict Fraud")

inputs = {}
for col in X.columns[:8]:
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"🚨 Fraud Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Claim (Fraud Probability: {prob:.2f})")
