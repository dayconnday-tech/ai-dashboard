import streamlit as st
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
# SAVE FILE
# =========================
if file:
    with open(DATA_PATH, "wb") as f:
        f.write(file.getbuffer())
    st.success("✅ New data uploaded & saved")

# =========================
# LOAD FILE
# =========================
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip().str.upper()
else:
    st.info("Upload Excel to start")
    st.stop()

st.subheader("📦 Latest Dataset")
st.dataframe(df.head())

st.write("📌 Columns detected:", df.columns.tolist())

# =========================
# SAFE COLUMN FINDER
# =========================
def find_col(df, keywords):
    for col in df.columns:
        if any(k in col for k in keywords):
            return col
    return None

# =========================
# CATEGORY LOGIC
# =========================
if category == "TV":

    price_col = find_col(df, ["PRIX", "PRICE"])
    size_col = find_col(df, ["POUCES", "SIZE", "INCH"])

    if not price_col or not size_col:
        st.error("Missing required columns for TV (PRIX / POUCES)")
        st.stop()

    df["price"] = pd.to_numeric(
        df[price_col].astype(str)
        .str.replace("Da", "", regex=False)
        .str.replace(" ", "", regex=False),
        errors="coerce"
    )

    df["feature"] = pd.to_numeric(
        df[size_col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    features = ["price", "feature"]

else:

    price_col = find_col(df, ["PRIX", "PRICE"])
    power_col = find_col(df, ["PUISSANCE", "POWER"])
    capacity_col = find_col(df, ["LITTRAGE", "CAPACITY"])
    speed_col = find_col(df, ["VITESSES", "SPEED"])

    if not price_col or not power_col or not capacity_col:
        st.error("Missing required PETRIN columns")
        st.stop()

    df["price"] = pd.to_numeric(
        df[price_col].astype(str).str.replace(" ", "", regex=False),
        errors="coerce"
    )

    df["power"] = pd.to_numeric(
        df[power_col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    df["capacity"] = pd.to_numeric(
        df[capacity_col].astype(str).str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )

    if speed_col:
        df["speeds"] = pd.to_numeric(
            df[speed_col].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce"
        )
    else:
        df["speeds"] = 0

    features = ["price", "power", "capacity", "speeds"]

# =========================
# CLEAN DATA
# =========================
df_clean = df.dropna(subset=features)

st.write("📊 Clean dataset size:", df_clean.shape)

if df_clean.empty:
    st.error("No valid numeric data for clustering")
    st.stop()

# =========================
# SAFE KMEANS
# =========================
n_clusters = min(4, len(df_clean))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df.loc[df_clean.index, "segment"] = model.fit_predict(X_scaled)

segment_map = {
    0: "Budget",
    1: "Mid Range",
    2: "Premium",
    3: "Flagship"
}

df["segment_name"] = df["segment"].map(segment_map)

# =========================
# KPI SECTION
# =========================
st.subheader("📊 KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("Total Products", len(df))

avg_price = df["price"].mean()
col2.metric(
    "Avg Price",
    f"{avg_price:.0f} DA" if pd.notna(avg_price) else "N/A"
)

top_segment = df["segment_name"].value_counts().idxmax()
col3.metric("Top Segment", top_segment)

# =========================
# TABLE
# =========================
st.subheader("📋 Live Data")
st.dataframe(df, use_container_width=True)

# =========================
# CHART
# =========================
st.subheader("📈 Segment Distribution")
st.bar_chart(df["segment_name"].value_counts().sort_values())

# =========================
# INSIGHT
# =========================
share = df["segment_name"].value_counts(normalize=True).max() * 100

st.success(
    f"{category}: {top_segment} dominates with {share:.1f}% of products"
)

# =========================
# DOWNLOAD
# =========================
st.download_button(
    "⬇️ Download Processed Data",
    df.to_csv(index=False),
    file_name="processed_data.csv",
    mime="text/csv"
)
