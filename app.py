import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("AI Product Dashboard System 🚀")

# ---------------------------
# CATEGORY
# ---------------------------
category = st.selectbox("Select Category", ["TV", "PETRIN"])

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file:

    df = pd.read_excel(file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # CLEAN COLUMN NAMES (VERY IMPORTANT FIX)
    # ---------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    st.write("Detected Columns:", df.columns.tolist())

    # =========================
    # TV LOGIC
    # =========================
    if category == "TV":

        try:
            df["price"] = (
                df["PRIX"]
                .astype(str)
                .str.replace("Da", "", regex=False)
                .str.replace(" ", "", regex=False)
                .astype(float)
            )

            df["feature"] = (
                df["POUCES"]
                .astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )

            feature_cols = ["price", "feature"]

        except:
            st.error("TV ERROR → check PRIX / POUCES columns")
            st.stop()

    # =========================
    # PETRIN LOGIC
    # =========================
    elif category == "PETRIN":

        try:
            # PRICE
            df["price"] = (
                df["PRIX"]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .astype(float)
            )

            # POWER
            df["power"] = (
                df["PUISSANCE"]
                .astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )

            # CAPACITY
            df["capacity"] = (
                df["LITTRAGE"]
                .astype(str)
                .str.extract(r"(\d+\.?\d*)")
                .astype(float)
            )

            # SPEEDS (optional safe)
            if "VITESSES" in df.columns:
                df["speeds"] = (
                    df["VITESSES"]
                    .astype(str)
                    .str.extract(r"(\d+)")
                    .astype(float)
                )
            else:
                df["speeds"] = 0

            feature_cols = ["price", "power", "capacity", "speeds"]

        except Exception as e:
            st.error(f"PETRIN ERROR → {e}")
            st.stop()

    # ---------------------------
    # CLEAN DATA
    # ---------------------------
    df_clean = df.dropna(subset=feature_cols)

    st.subheader("Cleaned Data")
    st.dataframe(df_clean[feature_cols].head())

    # ---------------------------
    # AI SEGMENTATION
    # ---------------------------
    model = KMeans(n_clusters=4, random_state=42)

    clusters = model.fit_predict(df_clean[feature_cols])

    df.loc[df_clean.index, "segment"] = clusters

    # ---------------------------
    # SEGMENT LABELS
    # ---------------------------
    segment_map = {
        0: "Budget",
        1: "Mid Range",
        2: "Premium",
        3: "Flagship"
    }

    df["segment_name"] = df["segment"].map(segment_map)

    # ---------------------------
    # OUTPUT
    # ---------------------------
    st.subheader("Segmented Data")
    st.dataframe(df)

    # ---------------------------
    # KPIs
    # ---------------------------
    st.subheader("KPIs")

    st.write("Total products:", len(df))
    st.write("Average price:", round(df["price"].mean(), 2))

    st.subheader("KPIs by Segment")
    st.dataframe(df.groupby("segment_name")["price"].agg(["count", "mean"]))

    # ---------------------------
    # CHARTS
    # ---------------------------
    st.subheader("Segment Distribution")
    st.bar_chart(df["segment_name"].value_counts())

    # ---------------------------
    # AI INSIGHT
    # ---------------------------
    st.subheader("AI Insight")

    top_segment = df["segment_name"].value_counts().idxmax()

    st.success(f"{category} dominant segment: {top_segment}")
