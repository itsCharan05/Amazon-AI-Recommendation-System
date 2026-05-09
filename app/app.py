import streamlit as st
import pandas as pd
import os
import time
import plotly.express as px
import plotly.graph_objects as go

from scipy.sparse.linalg import svds
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Amazon AI Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
/* ---- Base & Background ---- */
[data-testid="stAppViewContainer"] {
    background: #0a0f1e;
}
[data-testid="stSidebar"] {
    background: #0d1527 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stHeader"] {
    background: transparent;
}

/* ---- Typography ---- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #e2e8f0;
}

/* ---- Hero ---- */
.hero-wrap {
    background: linear-gradient(135deg, #1a1040 0%, #0f1e3d 60%, #0a1628 100%);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    padding: 36px 40px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 0;
}
.hero-badges {
    margin-top: 14px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}
.badge-purple { background: rgba(124,58,237,0.2); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.badge-blue   { background: rgba(37,99,235,0.2);  color: #60a5fa; border: 1px solid rgba(37,99,235,0.3); }
.badge-green  { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }

/* ---- Metric Cards ---- */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(124,58,237,0.4); }
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 4px;
}

/* ---- Glass Cards ---- */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 24px;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: white;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- Recommend Cards ---- */
.rec-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.2s, background 0.2s;
}
.rec-card:hover {
    border-color: rgba(124,58,237,0.4);
    background: rgba(124,58,237,0.07);
}
.rec-rank {
    font-size: 0.78rem;
    font-weight: 700;
    color: #7c3aed;
    min-width: 28px;
}
.rec-name {
    color: #e2e8f0;
    font-size: 0.92rem;
    font-weight: 600;
    flex: 1;
    padding: 0 12px;
}
.rec-rating {
    color: #facc15;
    font-size: 0.85rem;
    font-weight: 700;
    white-space: nowrap;
}
.rec-sim {
    color: #34d399;
    font-size: 0.85rem;
    font-weight: 700;
    white-space: nowrap;
}

/* ---- Leaderboard ---- */
.lb-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}
.lb-rank { font-size: 1.1rem; min-width: 30px; }
.lb-name { flex: 1; font-size: 0.9rem; color: #cbd5e1; font-weight: 500; }
.lb-stars { color: #facc15; font-size: 0.85rem; font-weight: 700; }
.lb-count { color: #64748b; font-size: 0.78rem; }

/* ---- Selectbox & Inputs ---- */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
}
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ---- Buttons ---- */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 13px 20px;
    font-size: 0.95rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    letter-spacing: 0.3px;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ---- Sidebar ---- */
.sidebar-section {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    margin-bottom: 10px;
}

/* ---- Tab styling ---- */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    padding: 24px 0 8px;
    color: #334155;
    font-size: 0.82rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 20px;
}

/* ---- Divider ---- */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(BASE_DIR, "data", "processed", "clean_data.csv")
    df = pd.read_csv(data_path)

    product_path = os.path.join(BASE_DIR, "data", "raw", "amazon-product-review.csv")
    products = pd.read_csv(product_path)[['asins', 'name', 'reviews.text']]
    products = products.drop_duplicates()
    products.columns = ['item_id', 'product_name', 'review_text']

    df = df.merge(products[['item_id', 'product_name']], on='item_id', how='left')
    df['product_name'] = df['product_name'].fillna("Unknown Product")
    df['product_name'] = df['product_name'].str.slice(0, 60)

    return df, products


df, products = load_data()
item_name_map = dict(zip(df['item_id'], df['product_name']))


# ================================
# TRAIN SVD MODEL (FIXED VERSION)
# ================================
@st.cache_resource
def train_model(df):

    # Unique users/items
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()

    # Index mappings
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}

    # Convert dataframe to sparse matrix indices
    rows = df['user_id'].map(user_idx)
    cols = df['item_id'].map(item_idx)

    from scipy.sparse import csr_matrix

    # Create sparse rating matrix
    matrix = csr_matrix(
        (df['rating'], (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    ).astype(np.float32)

    # Convert to dense for SVD
    matrix_dense = matrix.toarray()

    # Compute mean ratings ONLY on rated items
    user_ratings_mean = np.true_divide(
        matrix_dense.sum(axis=1),
        (matrix_dense != 0).sum(axis=1)
    )

    # Replace NaN values
    user_ratings_mean = np.nan_to_num(user_ratings_mean)

    # Mean center
    matrix_demeaned = matrix_dense - user_ratings_mean.reshape(-1, 1)

    # Keep missing values as 0
    matrix_demeaned[matrix_dense == 0] = 0

    # Number of latent factors
    k = min(50, min(matrix_dense.shape) - 1)

    # Perform SVD
    U, sigma, Vt = svds(matrix_demeaned, k=k)

    sigma = np.diag(sigma)

    # Reconstruct predicted ratings
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    # Add user means back
    predicted_ratings += user_ratings_mean.reshape(-1, 1)

    # Clamp ratings between 1 and 5
    predicted_ratings = np.clip(predicted_ratings, 1, 5)

    return predicted_ratings, user_idx, item_idx


predicted_ratings, user_idx, item_idx = train_model(df)


# ================================
# NLP MODEL
# ================================
@st.cache_resource
def build_nlp_model(products):
    review_df = products[['item_id', 'review_text']].dropna()
    grouped_reviews = review_df.groupby('item_id')['review_text'].apply(
        lambda x: " ".join(x)
    ).reset_index()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(grouped_reviews['review_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return grouped_reviews, cosine_sim

grouped_reviews, cosine_sim = build_nlp_model(products)

# ================================
# SVD RECOMMENDATIONS
# ================================
def recommend_svd(user_id, n=5):

    # Cold start handling
    if user_id not in user_idx:

        popular = (
            df.groupby('product_name')['rating']
            .mean()
            .sort_values(ascending=False)
            .head(n)
        )

        return [(name, round(score, 2)) for name, score in popular.items()]

    uidx = user_idx[user_id]

    # Products already rated by user
    user_items = set(df[df['user_id'] == user_id]['item_id'])

    # Predicted scores
    scores = np.nan_to_num(predicted_ratings[uidx])

    # Reverse item mapping
    idx_item = {v: k for k, v in item_idx.items()}

    predictions = []

    for i in range(len(scores)):

        item_id = idx_item[i]

        # Skip already rated items
        if item_id not in user_items:

            predictions.append((item_id, scores[i]))

    # Sort descending
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Convert item IDs to product names
    recommendations = [
        (
            item_name_map.get(item, "Unknown Product"),
            round(float(score), 2)
        )
        for item, score in predictions[:n]
    ]

    return recommendations

# ================================
# NLP RECOMMENDATIONS
# ================================
def recommend_nlp(item_id, n=5):
    if item_id not in grouped_reviews['item_id'].values:
        return []
    idx = grouped_reviews[grouped_reviews['item_id'] == item_id].index[0]
    similarity_scores = sorted(
        list(enumerate(cosine_sim[idx])),
        key=lambda x: x[1], reverse=True
    )[1:n+1]
    return [
        (item_name_map.get(grouped_reviews.iloc[i]['item_id'], "Unknown"), round(score, 3))
        for i, score in similarity_scores
    ]


# ================================
# METRICS
# ================================
total_users   = df['user_id'].nunique()
total_products = df['item_id'].nunique()
total_reviews  = len(df)
avg_rating     = round(df['rating'].mean(), 2)


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:2rem;'>🛒</div>
        <div style='font-weight:800; font-size:1rem; color:white; margin-top:4px;'>AI Recommender</div>
        <div style='color:#64748b; font-size:0.78rem;'>Amazon Products</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)

    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    st.divider()

    st.markdown('<div class="sidebar-title">📊 Dataset Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-section">
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='color:#94a3b8; font-size:0.82rem;'>Users</span>
            <span style='color:#a78bfa; font-weight:700; font-size:0.82rem;'>{total_users:,}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='color:#94a3b8; font-size:0.82rem;'>Products</span>
            <span style='color:#60a5fa; font-weight:700; font-size:0.82rem;'>{total_products:,}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='color:#94a3b8; font-size:0.82rem;'>Reviews</span>
            <span style='color:#34d399; font-weight:700; font-size:0.82rem;'>{total_reviews:,}</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='color:#94a3b8; font-size:0.82rem;'>Avg Rating</span>
            <span style='color:#facc15; font-weight:700; font-size:0.82rem;'>⭐ {avg_rating}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='color:#334155; font-size:0.75rem; text-align:center; padding-top:4px;'>
        SVD + TF-IDF · Hybrid Engine
    </div>
    """, unsafe_allow_html=True)


# ================================
# HERO
# ================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🛒 Amazon AI Recommendation System</div>
    <div class="hero-subtitle">Hybrid engine powered by Collaborative Filtering (SVD) and Content-Based NLP (TF-IDF)</div>
    <div class="hero-badges">
        <span class="badge badge-purple">✦ SVD Collaborative Filtering</span>
        <span class="badge badge-blue">✦ TF-IDF Content Similarity</span>
        <span class="badge badge-green">✦ Hybrid Recommendation</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ================================
# METRICS DASHBOARD
# ================================
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{total_users:,}</div>
        <div class="metric-label">👤 Total Users</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_products:,}</div>
        <div class="metric-label">📦 Products</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_reviews:,}</div>
        <div class="metric-label">💬 Reviews</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_rating}</div>
        <div class="metric-label">⭐ Avg Rating</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================================
# TABS
# ================================
tab1, tab2, tab3 = st.tabs(["🎯  Recommendations", "📊  Analytics", "🏆  Leaderboard"])


# ================================
# TAB 1 — RECOMMENDATIONS
# ================================
with tab1:
    col1, col2 = st.columns(2, gap="large")

    # --- LEFT: SVD ---
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Personalized Recommendations</div>', unsafe_allow_html=True)

        search_user = st.text_input("🔍 Search user ID", placeholder="Type to filter...", key="search_user")
        user_list = df['user_id'].unique()
        filtered_users = [u for u in user_list if search_user.lower() in str(u).lower()] if search_user else user_list
        selected_user = st.selectbox("Select User", filtered_users, key="user_sel")

        if st.button("✦ Get Recommendations", key="btn_svd"):
            with st.spinner("Running SVD model..."):
                time.sleep(0.6)
                recs = recommend_svd(selected_user, n=top_n)

            if recs:
                st.markdown(f"<div style='color:#64748b; font-size:0.8rem; margin-bottom:12px;'>Top {len(recs)} picks for <b style='color:#a78bfa'>{selected_user[:20]}…</b></div>", unsafe_allow_html=True)
                for i, (name, score) in enumerate(recs, 1):
                    medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
                    st.markdown(f"""
                    <div class="rec-card">
                        <span class="rec-rank">{medal}</span>
                        <span class="rec-name">{name}</span>
                        <span class="rec-rating">⭐ {score}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Export
                rec_df = pd.DataFrame(recs, columns=["Product", "Predicted Rating"])
                csv = rec_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Export CSV", csv, f"svd_recs_{selected_user[:10]}.csv", "text/csv", use_container_width=True)
            else:
                st.warning("No recommendations found for this user.")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- RIGHT: NLP ---
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 AI Product Recommendations</div>', unsafe_allow_html=True)

        search_prod = st.text_input("🔍 Search product", placeholder="Type to filter...", key="search_prod")
        product_list = df['product_name'].unique()
        filtered_prods = [p for p in product_list if search_prod.lower() in p.lower()] if search_prod else product_list
        selected_product = st.selectbox("Select Product", filtered_prods, key="prod_sel")

        if st.button("✦ Find Similar Products", key="btn_nlp"):
            with st.spinner("Computing TF-IDF similarity..."):
                time.sleep(0.6)
                item_id = df[df['product_name'] == selected_product]['item_id'].iloc[0]
                similar = recommend_nlp(item_id, n=top_n)

            if similar:
                st.markdown(f"<div style='color:#64748b; font-size:0.8rem; margin-bottom:12px;'>Top {len(similar)} products similar to <b style='color:#60a5fa'>{selected_product[:30]}…</b></div>", unsafe_allow_html=True)
                for i, (name, score) in enumerate(similar, 1):
                    medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
                    pct = int(score * 100)
                    st.markdown(f"""
                    <div class="rec-card">
                        <span class="rec-rank">{medal}</span>
                        <span class="rec-name">{name}</span>
                        <span class="rec-sim">~{pct}% match</span>
                    </div>
                    """, unsafe_allow_html=True)

                sim_df = pd.DataFrame(similar, columns=["Product", "Similarity Score"])
                csv2 = sim_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Export CSV", csv2, f"nlp_recs_{item_id[:10]}.csv", "text/csv", use_container_width=True)
            else:
                st.warning("No similar products found.")

        st.markdown('</div>', unsafe_allow_html=True)


# ================================
# TAB 2 — ANALYTICS
# ================================
with tab2:
    st.markdown("#### 📊 User Activity & Rating Analytics")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        # Rating distribution
        rating_counts = df['rating'].value_counts().sort_index().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig1 = px.bar(
            rating_counts, x='Rating', y='Count',
            title='Rating Distribution',
            color='Count',
            color_continuous_scale=['#2563eb', '#7c3aed', '#a78bfa'],
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.03)',
            font_color='#94a3b8',
            title_font_color='white',
            coloraxis_showscale=False,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8'),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # Reviews per product (top 10)
        top_reviewed = (
            df.groupby('product_name')['rating']
            .count()
            .reset_index()
            .rename(columns={'rating': 'Review Count'})
            .sort_values('Review Count', ascending=False)
            .head(10)
        )
        top_reviewed['product_name'] = top_reviewed['product_name'].str.slice(0, 30) + '…'
        fig2 = px.bar(
            top_reviewed, x='Review Count', y='product_name',
            orientation='h',
            title='Top 10 Most Reviewed Products',
            color='Review Count',
            color_continuous_scale=['#0f6e56', '#1D9E75', '#34d399'],
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.03)',
            font_color='#94a3b8',
            title_font_color='white',
            coloraxis_showscale=False,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8', title=''),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")

    with c3:
        # Avg rating per product (top 10 rated with ≥5 reviews)
        avg_by_product = (
            df.groupby('product_name')
            .agg(avg_rating=('rating', 'mean'), count=('rating', 'count'))
            .reset_index()
            .query('count >= 5')
            .sort_values('avg_rating', ascending=False)
            .head(10)
        )
        avg_by_product['product_name'] = avg_by_product['product_name'].str.slice(0, 28) + '…'
        avg_by_product['avg_rating'] = avg_by_product['avg_rating'].round(2)
        fig3 = px.bar(
            avg_by_product, x='avg_rating', y='product_name',
            orientation='h',
            title='Top 10 Highest Rated Products (≥5 reviews)',
            color='avg_rating',
            color_continuous_scale=['#854F0B', '#EF9F27', '#facc15'],
            range_color=[3.5, 5],
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.03)',
            font_color='#94a3b8',
            title_font_color='white',
            coloraxis_showscale=False,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8', range=[3, 5.2]),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8', title=''),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # User review count distribution
        user_review_counts = df.groupby('user_id')['rating'].count().reset_index()
        user_review_counts.columns = ['user_id', 'review_count']
        fig4 = px.histogram(
            user_review_counts, x='review_count',
            nbins=30,
            title='User Review Count Distribution',
            color_discrete_sequence=['#7c3aed'],
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.03)',
            font_color='#94a3b8',
            title_font_color='white',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8', title='Reviews per User'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont_color='#94a3b8', title='Users'),
            margin=dict(t=40, b=20, l=10, r=10),
            showlegend=False,
        )
        fig4.update_traces(marker_line_width=0)
        st.plotly_chart(fig4, use_container_width=True)


# ================================
# TAB 3 — LEADERBOARD
# ================================
with tab3:
    st.markdown("#### 🏆 Top Products Leaderboard")

    lb_col1, lb_col2 = st.columns([2, 1], gap="large")

    with lb_col1:
        leaderboard = (
            df.groupby('product_name')
            .agg(avg_rating=('rating', 'mean'), review_count=('rating', 'count'))
            .reset_index()
            .query('review_count >= 5')
            .sort_values('avg_rating', ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        leaderboard['avg_rating'] = leaderboard['avg_rating'].round(2)

        medals = ["🥇","🥈","🥉"] + [f"#{i}" for i in range(4, 11)]

        for i, row in leaderboard.iterrows():
            stars = "★" * int(round(row['avg_rating'])) + "☆" * (5 - int(round(row['avg_rating'])))
            st.markdown(f"""
            <div class="lb-row">
                <div class="lb-rank">{medals[i]}</div>
                <div class="lb-name">{row['product_name']}</div>
                <div class="lb-stars">{stars} {row['avg_rating']}</div>
                <div class="lb-count">{row['review_count']} reviews</div>
            </div>
            """, unsafe_allow_html=True)

        # Export leaderboard
        lb_csv = leaderboard.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Export Leaderboard CSV", lb_csv, "leaderboard.csv", "text/csv", use_container_width=True)

    with lb_col2:
        # Donut — rating breakdown of top products
        top_ids = leaderboard['product_name'].tolist()
        top_df = df[df['product_name'].isin(top_ids)]
        rating_share = top_df['rating'].value_counts().sort_index()
        fig5 = go.Figure(go.Pie(
            labels=[f"{r} ★" for r in rating_share.index],
            values=rating_share.values,
            hole=0.55,
            marker_colors=['#1D4ED8','#2563eb','#7c3aed','#a78bfa','#facc15'],
            textfont_color='white',
        ))
        fig5.update_layout(
            title='Rating Breakdown (Top Products)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            title_font_color='white',
            legend=dict(font_color='#94a3b8'),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig5, use_container_width=True)


# ================================
# FOOTER
# ================================
st.markdown("""
<div class="footer">
    Built with Streamlit &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; NLP &nbsp;·&nbsp; SVD &nbsp;·&nbsp; Plotly
</div>
""", unsafe_allow_html=True)
