import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# =========================
# PAGE CONFIG (IMPORTANT UI UPGRADE)
# =========================
st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🧠 AI Product Intelligence Dashboard")

# =========================
# SIDEBAR (CLEAN NAV)
# =========================
st.sidebar.header("Controls")

category = st.sidebar.selectbox("Select Category", ["TV", "PETRIN"])
file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

# =========================
# MAIN APP
# =========================
if file:

    df = pd.read_excel(file)

    df.columns = df.columns.astype(str).str.strip().str.upper()

    st.subheader("📦 Raw Data Preview")
    st.dataframe(df.head())

    # =========================
    # TV
    # =========================
    if category == "TV":

        df["price"] = (
            df["PRIX"].astype(str)
            .str.replace("Da", "", regex=False)
            .str.replace(" ", "", regex=False)
            .astype(float)
        )

        df["feature"] = (
            df["POUCES"].astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

        features = ["price", "feature"]

    # =========================
    # PETRIN
    # =========================
    else:

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
    df.loc[df_clean.index, "segment"] = model.fit_predict(df_clean[features])

    segment_map = {0: "Budget", 1: "Mid Range", 2: "Premium", 3: "Flagship"}
    df["segment_name"] = df["segment"].map(segment_map)

    # =========================
    # KPI CARDS (NEW UI)
    # =========================
    st.subheader("📊 KPIs")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", len(df))
    col2.metric("Average Price", f"{df['price'].mean():.0f} DA")
    col3.metric("Top Segment", df["segment_name"].value_counts().idxmax())

    # =========================
    # DATA TABLE
    # =========================
    st.subheader("📋 Segmented Data")
    st.dataframe(df, use_container_width=True)

    # =========================
    # CHARTS
    # =========================
    st.subheader("📈 Segment Distribution")
    st.bar_chart(df["segment_name"].value_counts())

    # =========================
    # INSIGHT BOX
    # =========================
    st.subheader("🧠 AI Insight")

    st.success(
        f"{category} market is dominated by "
        f"**{df['segment_name'].value_counts().idxmax()}** segment"
    )
