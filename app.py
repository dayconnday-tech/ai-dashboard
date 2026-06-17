import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("AI Product Dashboard System 🚀")

# ---------------------------
# CATEGORY (TV ONLY FOR NOW)
# ---------------------------
category = st.selectbox("Select Category", ["TV"])

# ---------------------------
# UPLOAD EXCEL
# ---------------------------
file = st.file_uploader("Upload TV Excel File", type=["xlsx"])

if file:

    df = pd.read_excel(file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # CLEAN DATA (TV ONLY)
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

    except Exception as e:
        st.error("Error in column names or format (PRIX / POUCES)")
        st.stop()

    st.subheader("Cleaned Data")
    st.dataframe(df[["price", "size"]].head())

    # ---------------------------
    # REMOVE MISSING VALUES
    # ---------------------------
    df_clean = df.dropna(subset=["price", "size"])

    # ---------------------------
    # AI SEGMENTATION (KMEANS)
    # ---------------------------
    features = df_clean[["price", "size"]]

    model = KMeans(n_clusters=4, random_state=42)
    clusters = model.fit_predict(features)

    df.loc[df_clean.index, "segment"] = clusters

    # ---------------------------
    # RESULTS
    # ---------------------------
    st.subheader("Segmented Data")
    st.dataframe(df)

    # ---------------------------
    # KPIs
    # ---------------------------
    st.subheader("KPIs")

    st.write("Total products:", len(df))
    st.write("Average price:", round(df["price"].mean(), 2))

    # ---------------------------
    # VISUALIZATION
    # ---------------------------
    st.subheader("Segment Distribution")

    st.bar_chart(df["segment"].value_counts())
