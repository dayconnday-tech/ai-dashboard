# =========================
# AI MODEL
# =========================

model = KMeans(n_clusters=4, random_state=42)
df.loc[df_clean.index, "cluster"] = model.fit_predict(df_clean[features])

# --------------------------------
# Dynamic Segment Naming
# --------------------------------

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
# KPI UI
# =========================

st.header("📊 KPI Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Products", len(df))
col2.metric("Brands", df["BRAND"].nunique() if "BRAND" in df.columns else 0)
col3.metric("Average Price", f"{df['price'].mean():,.0f} DA")
col4.metric("Max Price", f"{df['price'].max():,.0f} DA")

# =========================
# DATA TABLE
# =========================

st.header("📋 Product Database")
st.dataframe(df, use_container_width=True)

# =========================
# SEGMENT DISTRIBUTION
# =========================

st.header("📈 Market Segmentation")

st.bar_chart(
    df["segment_name"].value_counts()
)

# =========================
# BRAND ANALYSIS
# =========================

if "BRAND" in df.columns:

    st.header("🏷️ Brand Market Share")

    brand_share = (
        df["BRAND"]
        .value_counts()
        .reset_index()
    )

    brand_share.columns = ["Brand", "Products"]

    st.dataframe(brand_share)

    st.bar_chart(
        df["BRAND"].value_counts()
    )

# =========================
# MARKET STUDY
# =========================

st.header("📖 Étude de Marché")

total_products = len(df)

avg_price = round(df["price"].mean(), 0)
min_price = round(df["price"].min(), 0)
max_price = round(df["price"].max(), 0)

st.markdown(f"""
### Vue Générale

- Produits analysés : **{total_products}**
- Prix moyen : **{avg_price:,.0f} DA**
- Prix minimum : **{min_price:,.0f} DA**
- Prix maximum : **{max_price:,.0f} DA**
""")

# =========================
# SEGMENT REPORTS
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

    avg_seg_price = round(seg_df["price"].mean(), 0)

    min_seg_price = round(seg_df["price"].min(), 0)
    max_seg_price = round(seg_df["price"].max(), 0)

    market_share = round(
        len(seg_df) / len(df) * 100,
        1
    )

    if "BRAND" in seg_df.columns:

        top_brands = (
            seg_df["BRAND"]
            .value_counts()
            .head(3)
            .index
            .tolist()
        )

        top_brands = ", ".join(top_brands)

    else:
        top_brands = "N/A"

    st.markdown(f"""
## {seg}

- Market Share : **{market_share}%**
- Products : **{len(seg_df)}**
- Average Price : **{avg_seg_price:,.0f} DA**
- Price Range : **{min_seg_price:,.0f} → {max_seg_price:,.0f} DA**
- Dominant Brands : **{top_brands}**
""")

# =========================
# CATEGORY SPECIFIC INSIGHT
# =========================

st.header("🧠 AI Market Analysis")

if category == "TV":

    st.info("""
### Television Market Positioning

**Entrée de Gamme**
- Mainly 32" to 43"
- Standard LED TVs
- Dominated by local/value brands
- Focus on affordability

**Milieu de Gamme**
- 43" to 55"
- Smart TV + 4K UHD
- Mainstream consumer market

**Milieu-Haut de Gamme**
- 55" to 65"
- QLED / advanced image quality
- Premium entertainment positioning

**Haut de Gamme**
- Large screens
- OLED / Mini LED / flagship models
- High-performance segment
""")

elif category == "PETRIN":

    st.info("""
### Petrin Market Positioning

**Entrée de Gamme**
- 5L to 8L capacity
- Value-oriented products
- Suitable for household use

**Milieu de Gamme**
- Better capacity and power
- Balanced price/performance

**Milieu-Haut de Gamme**
- Strong motors
- Larger bowls
- Frequent-use customers

**Haut de Gamme**
- Premium brands
- Professional-oriented products
- Maximum performance and durability
""")
