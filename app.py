import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🧠 AI Product Intelligence Dashboard (SIMPLE MODE)")

# =========================
# GOOGLE SHEETS CSV LINKS
# =========================
TV_URL = "https://docs.google.com/spreadsheets/d/1zbt9FBVaMNRImI4V2nbqB0mSQ6NsVnb6APJwyznd7Bg/export?format=csv"
PETRAIN_URL = "https://docs.google.com/spreadsheets/d/1RuyIWbjSiysb1GAVLML6EzPqVZlrsfL6AbFDjok8mTU/export?format=csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.astype(str).str.strip().str.upper().str.replace(" ", "_")
    return df

tv_df = load_data(TV_URL)
petrain_df = load_data(PETRAIN_URL)

# =========================
# CATEGORY
# =========================
category = st.sidebar.selectbox("Select Category", ["TV", "PETRAIN", "ALL"])

if category == "TV":
    df = tv_df
elif category == "PETRAIN":
    df = petrain_df
else:
    df = pd.concat([tv_df, petrain_df], ignore_index=True)

st.subheader("📦 Dataset")
st.dataframe(df)

# =========================
# COLUMN FINDER
# =========================
def find_col(df, keywords):
    for col in df.columns:
        if any(k in col for k in keywords):
            return col
    return None

# =========================
# PRICE COLUMN
# =========================
price_col = find_col(df, ["PRIX", "PRICE"])

if not price_col:
    st.error("No PRICE column found")
    st.stop()

df["price"] = pd.to_numeric(df[price_col], errors="coerce")

# =========================
# FEATURES
# =========================
features = ["price"]

# TV feature
if category != "PETRAIN":
    size_col = find_col(df, ["POUCES", "SIZE", "INCH"])
    if size_col:
        df["feature"] = pd.to_numeric(
            df[size_col].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce"
        )
        features.append("feature")

# PETRAIN features
else:
    power_col = find_col(df, ["PUISSANCE", "POWER"])
    capacity_col = find_col(df, ["LITTRAGE", "CAPACITY"])

    if power_col:
        df["power"] = pd.to_numeric(df[power_col], errors="coerce")
        features.append("power")

    if capacity_col:
        df["capacity"] = pd.to_numeric(df[capacity_col], errors="coerce")
        features.append("capacity")

# =========================
# CLEAN DATA
# =========================
df_clean = df.dropna(subset=features)

if len(df_clean) < 3:
    st.error("Not enough data")
    st.stop()

# =========================
# AI CLUSTERING
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
col2.metric("Avg Price", f"{df['price'].mean():.0f} DA")
col3.metric("Top Segment", df["segment_name"].value_counts().idxmax())

# =========================
# CHARTS
# =========================
st.subheader("📦 Segment Distribution")
st.bar_chart(df["segment_name"].value_counts())

st.subheader("💰 Price Distribution")
st.bar_chart(df["price"].value_counts())

st.subheader("📈 AI View")
st.scatter_chart(df[features].dropna())

# =========================
# TOP PRODUCTS
# =========================
st.subheader("💎 Top Products")
st.dataframe(df.sort_values("price", ascending=False).head(10))

# =========================
# INSIGHT
# =========================
st.success(
    f"{category}: {df['segment_name'].value_counts().idxmax()} dominates market"
)
