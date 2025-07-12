import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# === PAGE CONFIG ===
st.set_page_config(page_title="Network Threat Analyzer", layout="wide")

page = st.sidebar.selectbox("🔀 Choose a Page", [
    "1️⃣ Attack Classification (XGBoost)",
    "2️⃣ Anomaly Detection (Random Forest)"
])

# === Classification Labels (for VPN dataset) ===
class_labels = [
    'Anonymizing VPN', 'Commercial VPN', 'Enterprise VPN',
    'Non-VPN', 'Tor', 'Social VPN'
]

# === UTILS ===
def show_uploaded(df):
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

def show_pred(df):
    st.subheader("📊 Prediction Results")
    st.dataframe(df)
    st.download_button(
        "📥 Download Results",
        df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

# === PAGE 1: XGBoost Classification ===
if page.startswith("1️⃣"):
    st.title("🧠 Attack Classification (XGBoost)")

    @st.cache_resource
    def load_xgb():
        booster = xgb.Booster()
        booster.load_model("traffic-classification/xgb_model.json")
        return booster

    @st.cache_resource
    def load_label_encoder():
        df_train = pd.read_csv("traffic-classification/scenario_a_combined.csv")
        le = LabelEncoder()
        le.fit(df_train["class1"])
        return le

    booster = load_xgb()
    le = load_label_encoder()

    uploaded = st.file_uploader("📂 Upload CSV for Classification", type="csv", key="xgb")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head())

            # Drop non-numeric and class1 column if present
            df_num = df.select_dtypes(include=[np.number])
            if "class1" in df:
                df_num = df_num.drop(columns=["class1"], errors="ignore")

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df_num.values)

            dmat = xgb.DMatrix(X_scaled)
            preds = booster.predict(dmat).astype(int)

            df["Predicted Category"] = le.inverse_transform(preds)

            show_pred(df)

        except Exception as e:
            st.error(f"❌ Classification Error: {e}")

# === PAGE 2: RandomForest Anomaly Detection ===
else:
    st.title("🚨 Anomaly Detection (Random Forest)")

    @st.cache_resource
    def load_rf():
        rf = joblib.load("network-anomaly/model_plain_rf.pkl")
        scaler = joblib.load("network-anomaly/scaler_plain.pkl")
        return rf, scaler

    rf_model, rf_scaler = load_rf()

    # Attack Category Mapping (0-13)
    label_mapping = {
        0: "Analysis",
        1: "Backdoor",
        2: "DoS",
        3: "Exploits",
        4: "Fuzzers",
        5: "Generic",
        6: "Reconnaissance",
        7: "Shellcode",
        8: "Worms",
        9: "Shellcode",
        10: "Reconnaissance",
        11: "Fuzzers",
        12: "Worms",
        13: "Normal ✅"
    }

    uploaded = st.file_uploader("📂 Upload UNSW-NB15 CSV (no headers)", type="csv", key="rf")
    if uploaded:
        try:
            df = pd.read_csv(uploaded, header=None)
            show_uploaded(df)

            # Drop last two columns (label + attack_cat)
            ncol = df.shape[1]
            df = df.drop(columns=[ncol - 1, ncol - 2])

            df_num = df.select_dtypes(include=[np.number])

            X = df_num.values
            Xs = rf_scaler.transform(X)
            preds = rf_model.predict(Xs)

            # Map predictions to attack category names
            df["Anomaly Category"] = [label_mapping.get(i, "Unknown") for i in preds]

            show_pred(df)

        except Exception as e:
            st.error(f"❌ Detection Error: {e}")
