import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("AI Product Dashboard System 🚀")

# Only TV for now
category = st.selectbox("Select Category", ["TV"])

file = st.file_uploader("Upload TV Excel File", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ---------------------------
    # CLEAN TV DATA
    # ---------------------------
    try:
        df["price"] = df["PRIX"].astype(str).str.replace("Da", "").str.replace(" ", "").astype(float)
        df["size"] = df["POUCES"].astype(str).str.extract(r"(\d+)").astype(float)
    except:
        st.error("Check column names: PRIX / POUCES")
        st.stop()

    st.subheader("Cleaned Data")
    st.dataframe(df[["price", "size"]].head())

    # ---------------------------
    # AI SEGMENTATION (KMEANS)
    # ---------------------------
    features = df[["price", "size"]].dropna()

    model = KMeans(n_clusters=4, random_state=42)
    df["segment"] = model.fit_predict(features)

    st.subheader("Segmented Data")
    st.dataframe(df)

    # ---------------------------
    # KPIs
    # ---------------------------
    st.subheader("KPIs")

    st.write("Total products:", len(df))
    st.write("Average price:", df["price"].mean())

    st.bar_chart(df["segment"].value_counts())
