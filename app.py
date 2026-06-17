import streamlit as st
import pandas as pd
import os
from sklearn.cluster import KMeans

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🧠 AI Product Intelligence Dashboard")

# =========================
# CATEGORY
# =========================
category = st.sidebar.selectbox("Select Category", ["TV", "PETRIN"])

# =========================
# FILE UPLOAD
# =========================
file = st.sidebar.file_uploader("Upload NEW Excel (updates system)", type=["xlsx"])

DATA_PATH = "latest_data.xlsx"

# =========================
# SAVE LATEST FILE (IMPORTANT)
# =========================
if file:
    with open(DATA_PATH, "wb") as f:
        f.write(file.getbuffer())
    st.success("✅ New data uploaded & saved")

# =========================
# LOAD LATEST FILE
# =========================
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip().str.upper()
else:
    st.info("Upload Excel to start")
    st.stop()

# =========================
# SHOW DATA
# =========================
st.subheader("📦 Latest Dataset")
st.dataframe(df.head())

# =========================
# TV LOGIC
# =========================
if category == "TV":

    df["price"] = (
        df["PRIX"].astype(str)
        .str.replace("Da", "", regex=False)
        .str.replace(" ", "", regex=False)
        .astype(float)
    )

    df["feature"] = df["POUCES"].astype(str).str.extract(r"(\d+)").astype(float)

    features = ["price", "feature"]

# =========================
# PETRIN LOGIC
# =========================
else:

    df["price"] = df["PRIX"].astype(str).str.replace(" ", "", regex=False).astype(float)
    df["power"] = df["PUISSANCE"].astype(str).str.extract(r"(\d+)").astype(float)
    df["capacity"] = df["LITTRAGE"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    if "VITESSES" in df.columns:
        df["speeds"] = df["VITESSES"].astype(str).str.extract(r"(\d+)").astype(float)
    else:
        df["speeds"] = 0

    features = ["price", "power", "capacity", "speeds"]

# =========================
# CLEAN
# =========================
df_clean = df.dropna(subset=features)

# =========================
# AI MODEL
# =========================
model = KMeans(n_clusters=4, random_state=42)
df.loc[df_clean.index, "segment"] = model.fit_predict(df_clean[features])

segment_map = {0: "Budget", 1: "Mid Range", 2: "Premium", 3: "Flagship"}
df["segment_name"] = df["segment"].map(segment_map)

# =========================
# KPI UI
# =========================
st.subheader("📊 KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("Total Products", len(df))
col2.metric("Avg Price", f"{df['price'].mean():.0f} DA")
col3.metric("Top Segment", df["segment_name"].value_counts().idxmax())

# =========================
# TABLE
# =========================
st.subheader("📋 Live Data")
st.dataframe(df, use_container_width=True)

# =========================
# CHART
# =========================
st.subheader("📈 Segment Distribution")
st.bar_chart(df["segment_name"].value_counts())

# =========================
# INSIGHT
# =========================
st.success(
    f"{category} is dominated by "
    f"{df['segment_name'].value_counts().idxmax()} segment"
)
