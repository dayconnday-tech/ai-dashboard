import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("AI Product Dashboard System 🚀")

# ---------------------------
# CATEGORY SELECTION
# ---------------------------
category = st.selectbox("Select Category", ["TV", "PETRIN"])

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file:

    df = pd.read_excel(file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # CATEGORY LOGIC
    # ---------------------------
    try:

        # ================= TV =================
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

            feature_cols = ["price", "feature"]

        # ================= PETRIN =================
        elif category == "PETRIN":

            df["price"] = (
                df["PRIX"].astype(str)
                .str.replace("Da", "", regex=False)
                .str.replace(" ", "", regex=False)
                .astype(float)
            )

            df["power"] = (
                df["PUISSANCE"].astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )

            df["capacity"] = (
                df["LITRAGE"].astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )

            feature_cols = ["price", "power", "capacity"]

    except:
        st.error("Check your Excel column names (PRIX / POUCES / PUISSANCE / LITRAGE)")
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
