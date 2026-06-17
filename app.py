import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("AI Product Dashboard System 🚀")

# ---------------------------
# CATEGORY (TV ONLY FOR NOW)
# ---------------------------
category = st.selectbox("Select Category", ["TV"])

# ---------------------------
# UPLOAD FILE
# ---------------------------
file = st.file_uploader("Upload TV Excel File", type=["xlsx"])

if file:

    df = pd.read_excel(file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # CLEAN DATA
    # ---------------------------
    try:
        df["price"] = (
            df["PRIX"]
            .astype(str)
            .str.replace("Da", "", regex=False)
            .str.replace(" ", "", regex=False)
            .astype(float)
        )

        df["size"] = (
            df["POUCES"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    except:
        st.error("Check column names: PRIX / POUCES")
        st.stop()

    st.subheader("Cleaned Data")
    st.dataframe(df[["price", "size"]].head())

    # ---------------------------
    # REMOVE NULLS
    # ---------------------------
    df_clean = df.dropna(subset=["price", "size"])

    # ---------------------------
    # AI SEGMENTATION
    # ---------------------------
    features = df_clean[["price", "size"]]

    model = KMeans(n_clusters=4, random_state=42)
    clusters = model.fit_predict(features)

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
    # FINAL DATA
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

    st.success(f"Dominant market segment: {top_segment}")
