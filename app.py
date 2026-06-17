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
file = st.sidebar.file_uploader("Upload NEW Excel", type=["xlsx"])

DATA_PATH = "latest_data.xlsx"

if file:
    with open(DATA_PATH, "wb") as f:
        f.write(file.getbuffer())
    st.success("✅ New data uploaded & saved")

# =========================
# SMART EXCEL LOADER (ROBUST)
# =========================
def load_excel(path):
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(how="all")

    # Fix broken headers (Unnamed columns issue)
    if df.columns.astype(str).str.contains("Unnamed").all():
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        df = df.dropna(how="all")

        for i in range(min(10, len(df))):
            if df.iloc[i].notna().sum() > 2:
                df.columns = df.iloc[i]
                df = df[i + 1:]
                break

    df = df.reset_index(drop=True)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    return df

# =========================
# LOAD DATA
# =========================
if os.path.exists(DATA_PATH):
    df = load_excel(DATA_PATH)
else:
    st.info("Upload Excel to start")
    st.stop()

st.subheader("📦 Dataset Preview")
st.dataframe(df.head())

st.write("📌 Columns detected:", df.columns.tolist())

# =========================
# COLUMN FINDER
# =========================
def find_col(df, keywords):
    for col in df.columns:
        if any(k in col for k in keywords):
            return col
    return None

# =========================
# TV LOGIC
# =========================
if category == "TV":

    price_col = find_col(df, ["PRIX", "PRICE"])
    size_col = find_col(df, ["POUCES", "INCH", "SIZE"])

    if not price_col or not size_col:
        st.error("Missing TV columns (PRIX / POUCES)")
        st.stop()

    df["price"] = pd.to_numeric(
        df[price_col].astype(str)
        .str.replace("DA", "", regex=False)
        .str.replace(" ", "", regex=False),
        errors="coerce"
    )

    df["feature"] = pd.to_numeric(
        df[size_col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    features = ["price", "feature"]

# =========================
# PETRIN LOGIC
# =========================
else:

    price_col = find_col(df, ["PRIX", "PRICE"])
    power_col = find_col(df, ["PUISSANCE", "POWER"])
    capacity_col = find_col(df, ["LITTRAGE", "CAPACITY"])
    speed_col = find_col(df, ["VITESSES", "SPEED"])

    if not price_col or not power_col or not capacity_col:
        st.error("Missing PETRIN columns")
        st.stop()

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")

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

if len(df_clean) < 3:
    st.error("Not enough data for clustering (min 3 rows required)")
    st.stop()

# =========================
# KMEANS
# =========================
n_clusters = max(2, min(4, len(df_clean)))

scaler = StandardScaler()
X = scaler.fit_transform(df_clean[features])

model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df.loc[df_clean.index, "segment"] = model.fit_predict(X)

segment_map = {
    0: "Budget",
    1: "Mid Range",
    2: "Premium",
    3: "Flagship"
}

df["segment_name"] = df["segment"].map(segment_map)

# =========================
# KPIs
# =========================
st.subheader("📊 KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("Total Products", len(df))

avg_price = df["price"].mean()
col2.metric("Avg Price", f"{avg_price:.0f} DA" if pd.notna(avg_price) else "N/A")

if df["segment_name"].notna().any():
    top_segment = df["segment_name"].value_counts().idxmax()
else:
    top_segment = "Unknown"

col3.metric("Top Segment", top_segment)

# =========================
# TABLE
# =========================
st.subheader("📋 Live Data")
st.dataframe(df, use_container_width=True)

# =========================
# CHART 1: SEGMENT DISTRIBUTION
# =========================
st.subheader("📦 Products per Segment")
st.bar_chart(df["segment_name"].value_counts().sort_values())

# =========================
# CHART 2: PRICE DISTRIBUTION
# =========================
st.subheader("💰 Price Distribution")
st.bar_chart(df["price"].dropna().value_counts().sort_index())

# =========================
# CHART 3: AVG PRICE BY SEGMENT
# =========================
st.subheader("📊 Avg Price by Segment")
st.bar_chart(df.groupby("segment_name")["price"].mean().sort_values())

# =========================
# CHART 4: AI SCATTER VIEW
# =========================
st.subheader("📈 AI Cluster View")

if category == "TV":
    st.scatter_chart(df[["price", "feature"]].dropna())
else:
    cols = [c for c in ["price", "power", "capacity", "speeds"] if c in df.columns]
    if len(cols) >= 2:
        st.scatter_chart(df[cols].dropna())

# =========================
# TOP PRODUCTS
# =========================
st.subheader("💎 Top 10 Expensive Products")
st.dataframe(df.sort_values("price", ascending=False).head(10))

# =========================
# SIMPLE PIE (NO DEPENDENCIES)
# =========================
st.subheader("🥧 Market Share (Simple)")

st.bar_chart(df["segment_name"].value_counts())

# =========================
# INSIGHT
# =========================
share = df["segment_name"].value_counts(normalize=True).max() * 100

st.success(f"{category}: {top_segment} dominates with {share:.1f}% share")

# =========================
# DOWNLOAD
# =========================
st.download_button(
    "⬇️ Download Processed Data",
    df.to_csv(index=False),
    file_name="processed_data.csv",
    mime="text/csv"
)
