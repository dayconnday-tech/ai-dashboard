import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🧠 AI Product Intelligence Dashboard (LIVE GOOGLE SHEETS)")

# =========================
# GOOGLE SHEETS AUTH
# =========================
@st.cache_data
def connect_gsheets():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_file(
        "credentials.json",
        scopes=scope
    )

    client = gspread.authorize(creds)
    return client

client = connect_gsheets()

# =========================
# LOAD TV SHEET
# =========================
@st.cache_data
def load_tv():
    sheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/1zbt9FBVaMNRImI4V2nbqB0mSQ6NsVnb6APJwyznd7Bg"
    ).sheet1

    df = pd.DataFrame(sheet.get_all_records())

    df.columns = df.columns.astype(str).str.strip().str.upper().str.replace(" ", "_")

    df["source"] = "TV"
    return df

# =========================
# LOAD PETRAIN SHEET
# =========================
@st.cache_data
def load_petrain():
    sheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/1RuyIWbjSiysb1GAVLML6EzPqVZlrsfL6AbFDjok8mTU"
    ).sheet1

    df = pd.DataFrame(sheet.get_all_records())

    df.columns = df.columns.astype(str).str.strip().str.upper().str.replace(" ", "_")

    df["source"] = "PETRAIN"
    return df

# =========================
# SIDEBAR SELECT
# =========================
category = st.sidebar.selectbox("Select Category", ["TV", "PETRAIN", "ALL"])

# =========================
# LOAD DATA
# =========================
if category == "TV":
    df = load_tv()
elif category == "PETRAIN":
    df = load_petrain()
else:
    df = pd.concat([load_tv(), load_petrain()], ignore_index=True)

st.subheader("📦 Live Dataset")
st.dataframe(df)

st.write("📌 Columns:", df.columns.tolist())

# =========================
# SAFE COLUMN FINDER
# =========================
def find_col(df, keywords):
    for col in df.columns:
        if any(k in col for k in keywords):
            return col
    return None

# =========================
# COMMON AI FEATURES
# =========================
price_col = find_col(df, ["PRIX", "PRICE"])

if not price_col:
    st.error("No PRICE column found in sheet")
    st.stop()

df["price"] = pd.to_numeric(df[price_col], errors="coerce")

# =========================
# EXTRA FEATURES
# =========================
if category != "PETRAIN":

    size_col = find_col(df, ["POUCES", "INCH", "SIZE"])
    if size_col:
        df["feature"] = pd.to_numeric(
            df[size_col].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce"
        )

    features = ["price"]

    if "feature" in df.columns:
        features.append("feature")

else:

    power_col = find_col(df, ["PUISSANCE", "POWER"])
    capacity_col = find_col(df, ["LITTRAGE", "CAPACITY"])

    if power_col:
        df["power"] = pd.to_numeric(df[power_col], errors="coerce")

    if capacity_col:
        df["capacity"] = pd.to_numeric(df[capacity_col], errors="coerce")

    features = ["price"]

    if "power" in df.columns:
        features.append("power")
    if "capacity" in df.columns:
        features.append("capacity")

# =========================
# CLEAN DATA
# =========================
df_clean = df.dropna(subset=features)

st.write("📊 Clean size:", df_clean.shape)

if len(df_clean) < 3:
    st.error("Not enough data for AI clustering")
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
# KPI
# =========================
st.subheader("📊 KPIs")

col1, col2, col3 = st.columns(3)

col1.metric("Total Products", len(df))
col2.metric("Avg Price", f"{df['price'].mean():.0f} DA")
col3.metric(
    "Top Segment",
    df["segment_name"].value_counts().idxmax()
)

# =========================
# CHARTS
# =========================
st.subheader("📦 Segment Distribution")
st.bar_chart(df["segment_name"].value_counts())

st.subheader("💰 Price Distribution")
st.bar_chart(df["price"].value_counts())

st.subheader("📈 AI View")
if len(features) >= 2:
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
