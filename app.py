import streamlit as st
import pandas as pd
import os
from sklearn.cluster import KMeans

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🧠 AI Product Intelligence Dashboard")

# =========================
# SIDEBAR
# =========================
category = st.sidebar.selectbox("Select Category", ["TV", "PETRIN"])
file = st.sidebar.file_uploader("Upload NEW Excel (updates system)", type=["xlsx"])

# =========================
# SAVE LATEST FILE
# =========================
DATA_PATH = "latest_data.xlsx"

if file:
    with open(DATA_PATH, "wb") as f:
        f.write(file.getbuffer())
    st.success("✅ New data uploaded & saved")

# =========================
# LOAD DATA
# =========================
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip().str.upper()
else:
    st.info("Upload Excel file to start")
    st.stop()

# =========================
# SHOW RAW DATA
# =========================
st.subheader("📦 Latest Dataset")
st.dataframe(df.head())

# =========================
# FEATURES BY CATEGORY
# =========================
if category == "TV":

    df["price"] = (
        df["PRIX"].astype(str)
        .str.replace("DA", "", regex=False)
        .str.replace(" ", "", regex=False)
        .astype(float)
    )

    df["feature"] = (
        df["POUCES"].astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    features = ["price", "feature"]

else:  # PETRIN

    df["price"] = (
        df["PRIX"].astype(str)
        .str.replace(" ", "", regex=False)
        .astype(float)
    )

    df["power"] = df["PUISSANCE"].astype(str).str.extract(r"(\d+)").astype(float)
    df["capacity"] = df["LITTRAGE"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    if "VITESSES" in df.columns:
        df["speeds"] = df["VITESSES"].astype(str).str.extract(r"(\d+)").astype(float)
    else:
        df["speeds"] = 0

    features = ["price", "power", "capacity", "speeds"]

# =========================
# CLEAN DATA
# =========================
df_clean = df.dropna(subset=features)

# =========================
# AI SEGMENTATION
# =========================
model = KMeans(n_clusters=4, random_state=42)
df.loc[df_clean.index, "cluster"] = model.fit_predict(df_clean[features])

# ORDER CLUSTERS BY PRICE (IMPORTANT FIX)
cluster_order = (
    df.groupby("cluster")["price"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

segment_map = {
    cluster_order[0]: "Entrée de Gamme",
    cluster_order[1]: "Milieu de Gamme",
    cluster_order[2]: "Milieu-Haut de Gamme",
    cluster_order[3]: "Haut de Gamme"
}

df["segment_name"] = df["cluster"].map(segment_map)

# =========================
# KPI DASHBOARD
# =========================
st.header("📊 KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("Products", len(df))
col2.metric("Avg Price", f"{df['price'].mean():.0f} DA")
col3.metric("Top Segment", df["segment_name"].value_counts().idxmax())

# =========================
# DATA TABLE
# =========================
st.header("📋 Live Data")
st.dataframe(df, use_container_width=True)

# =========================
# SEGMENT CHART
# =========================
st.header("📈 Market Segmentation")

st.bar_chart(df["segment_name"].value_counts())

# =========================
# BRAND ANALYSIS
# =========================
if "BRAND" in df.columns:

    st.header("🏷️ Brand Analysis")

    st.dataframe(df["BRAND"].value_counts().reset_index().rename(columns={
        "index": "Brand",
        "BRAND": "Products"
    }))

    st.bar_chart(df["BRAND"].value_counts())

# =========================
# MARKET STUDY
# =========================
st.header("📖 Étude de Marché")

total = len(df)
avg_price = df["price"].mean()
min_price = df["price"].min()
max_price = df["price"].max()

st.markdown(f"""
### Vue Générale

- Produits analysés : **{total}**
- Prix moyen : **{avg_price:,.0f} DA**
- Prix minimum : **{min_price:,.0f} DA**
- Prix maximum : **{max_price:,.0f} DA**
""")

# =========================
# SEGMENT REPORT
# =========================
for seg in [
    "Entrée de Gamme",
    "Milieu de Gamme",
    "Milieu-Haut de Gamme",
    "Haut de Gamme"
]:

    seg_df = df[df["segment_name"] == seg]

    if len(seg_df) == 0:
        continue

    st.markdown(f"""
## {seg}

- Produits : {len(seg_df)}
- Part de marché : {round(len(seg_df)/len(df)*100,1)}%
- Prix moyen : {seg_df['price'].mean():.0f} DA
- Prix min → max : {seg_df['price'].min():.0f} → {seg_df['price'].max():.0f} DA
""")

# =========================
# AI INSIGHT (FINAL MARKET VIEW)
# =========================
st.header("🧠 AI Market Insight")

st.success(f"""
The {category} market is mainly driven by the **{df['segment_name'].value_counts().idxmax()}** segment.

This indicates that the market is concentrated in mid-level affordability products,
with clear opportunities in underrepresented price ranges.
""")
