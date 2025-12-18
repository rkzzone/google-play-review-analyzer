"""
App Review Sentiment & Topic Intelligence Dashboard
Production-ready Streamlit application for analyzing Google Play Store reviews
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils import (
    search_app_hybrid,
    scrape_app_reviews,
    load_sentiment_model,
    predict_sentiment_batch,
    generate_topics,
    get_topic_labels,
    extract_ngrams,
    aggregate_by_date,
    filter_by_sentiment
)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="App Review Intelligence Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .review-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================

if 'reviews_df' not in st.session_state:
    st.session_state.reviews_df = None
if 'selected_app' not in st.session_state:
    st.session_state.selected_app = None
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
if 'topic_labels' not in st.session_state:
    st.session_state.topic_labels = {}

# =============================================================================
# Main Title
# =============================================================================

st.title("📱 App Review Sentiment & Topic Intelligence Dashboard")
st.markdown("**Analyze Google Play Store reviews with AI-powered sentiment analysis and topic modeling**")

# =============================================================================
# Sidebar - Control Panel
# =============================================================================

with st.sidebar:
    st.header("🎛️ Control Panel")
    st.info("Controls and filters will appear here after analysis")
    
    # Global filter (post-analysis)
    if st.session_state.reviews_df is not None:
        st.markdown("---")
        st.subheader("🔍 Filters")
        
        sentiment_filter = st.multiselect(
            "Filter by Sentiment:",
            options=['Positive', 'Neutral', 'Negative'],
            default=['Positive', 'Neutral', 'Negative'],
            help="Show only selected sentiment categories"
        )

# =============================================================================
# Main Content Area
# =============================================================================

if st.session_state.reviews_df is None:
    # Welcome screen with App Selection
    st.markdown("Search for an app to analyze its reviews with AI-powered sentiment analysis")
    
    st.subheader("1️⃣ Search for App")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Input tunggal yang menangani segala jenis input
        query = st.text_input(
            "Cari Aplikasi",
            placeholder="Nama aplikasi atau Link Play Store",
            label_visibility="collapsed",
            key="app_search_query"
        )
    with col2:
        search_clicked = st.button("🔍 Cari", use_container_width=True, type="primary")
    
    if search_clicked:
        if query:
            with st.spinner("Sedang mencari di Play Store..."):
                # Menggunakan fungsi hybrid yang sudah kita buat
                results = search_app_hybrid(query)
                
                if results:
                    st.session_state.search_results = results
                    st.success(f"✅ Ditemukan {len(results)} aplikasi!")
                else:
                    st.error("❌ Aplikasi tidak ditemukan. Pastikan ejaan benar atau coba paste Link Play Store-nya.")
        else:
            st.warning("⚠️ Masukkan kata kunci pencarian.")

    # --- BAGIAN DISPLAY HASIL (Di luar Tabs agar layout lebar penuh) ---
    if 'search_results' in st.session_state and st.session_state.search_results:
        st.markdown("---")
        st.subheader("2️⃣ Pilih Aplikasi dari Hasil Pencarian")
        
        # Konfigurasi Grid (5 Kolom, max 10 hasil = 2 baris)
        apps_per_row = 5
        results = st.session_state.search_results[:10]  # Limit to 10 results
        
        # Loop Dinamis: Menangani berapapun jumlah baris secara otomatis
        for i in range(0, len(results), apps_per_row):
            # Mengambil potongan data (chunk) untuk baris ini
            row_apps = results[i:i+apps_per_row]
            cols = st.columns(apps_per_row)
            
            for idx, app in enumerate(row_apps):
                with cols[idx]:
                    # Container dengan border (Kartu)
                    with st.container(border=True):
                        # Layout Icon di kiri, Teks di kanan
                        c_img, c_txt = st.columns([1, 3])
                        
                        with c_img:
                            if app.get('icon'):
                                st.image(app['icon'], use_container_width=True)
                            else:
                                st.markdown("<div style='text-align: center; font-size: 40px;'>📱</div>", unsafe_allow_html=True)
                        
                        with c_txt:
                            # Title dengan truncate untuk 5 kolom - max 15 karakter agar 1 baris
                            title = app['title']
                            max_title_len = 15
                            display_title = title[:max_title_len] + "..." if len(title) > max_title_len else title
                            st.markdown(f"**{display_title}**", help=title)
                            
                            # Info Developer & Rating - single line dengan truncate lebih ketat
                            score = app.get('score', 0)
                            rating_display = f"{score:.2f}" if score else "N/A"
                            dev = app.get('developer', 'Unknown')
                            # Max 8 karakter untuk developer karena space lebih sempit
                            max_dev_len = 8
                            display_dev = dev[:max_dev_len] + "..." if len(dev) > max_dev_len else dev
                            st.caption(f"⭐ {rating_display} | 👨‍💻 {display_dev}")

                        # Tombol Select di bawah (full width)
                        unique_key = f"btn_{app['appId']}_{i}_{idx}"
                        if st.button("Analisis", key=unique_key, use_container_width=True, type="secondary"):
                            st.session_state.selected_app = app
                            st.toast(f"Memilih aplikasi: {app['title']}")
                            st.rerun()
    
    # Scraping Configuration (after app selected)
    if st.session_state.selected_app:
        st.markdown("---")
        st.subheader("3️⃣ Configure Analysis")
        
        selected_app = st.session_state.selected_app
        st.info(f"**Selected App:** {selected_app['title']} by {selected_app.get('developer', 'Unknown')}")
        
        # Filter mode selection
        col1, col2 = st.columns(2)
        
        with col1:
            filter_mode = st.radio(
                "**Select Scraping Mode:**",
                options=["Review Count Limit", "Date Range"],
                help="Choose how to filter reviews"
            )
        
        with col2:
            # Conditional inputs based on mode
            if filter_mode == "Review Count Limit":
                review_count = st.number_input(
                    "Number of Reviews",
                    min_value=50,
                    max_value=5000,
                    value=500,
                    step=50,
                    help="Number of most recent reviews to fetch"
                )
                start_date = None
                end_date = None
            else:
                st.markdown("**Date Range:**")
                # Date inputs side by side
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=90),
                        max_value=datetime.now()
                    )
                with date_col2:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now(),
                        max_value=datetime.now()
                    )
                review_count = None
        
        st.markdown("---")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
                app_id = selected_app['appId']
                
                # Default to English language and US country
                lang = 'en'
                country = 'us'
                
                # Scrape reviews
                if filter_mode == "Review Count Limit":
                    reviews_df = scrape_app_reviews(
                        app_id=app_id,
                        lang=lang,
                        country=country,
                        filter_mode='count',
                        target_count=review_count
                    )
                else:
                    reviews_df = scrape_app_reviews(
                        app_id=app_id,
                        lang=lang,
                        country=country,
                        filter_mode='date_range',
                        start_date=pd.to_datetime(start_date),
                        end_date=pd.to_datetime(end_date)
                    )
                
                if not reviews_df.empty:
                    # Load sentiment model
                    model, tokenizer, device = load_sentiment_model()
                    
                    if model is not None:
                        # Predict sentiment
                        with st.spinner("🤖 Analyzing sentiment..."):
                            predictions, probabilities = predict_sentiment_batch(
                                reviews_df['review_text'].tolist(),
                                model,
                                tokenizer,
                                device
                            )
                            reviews_df['predicted_sentiment'] = predictions
                        
                        # Generate topics
                        with st.spinner("📊 Discovering topics..."):
                            topics, topic_model, topic_info = generate_topics(
                                reviews_df['review_text'].tolist()
                            )
                            
                            if topics is not None:
                                reviews_df['topic'] = topics
                                st.session_state.topic_model = topic_model
                                st.session_state.topic_labels = get_topic_labels(topic_model, topic_info)
                        
                        # Store in session state
                        st.session_state.reviews_df = reviews_df
                        st.success("✅ Analysis complete!")
                        st.rerun()
                else:
                    st.error("❌ Failed to fetch reviews. Please try again.")
    
    # Feature highlights (only show if no app selected)
    if not st.session_state.selected_app:
        st.markdown("---")
        st.markdown("## ✨ Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🎯 Sentiment Analysis")
            st.write("AI-powered sentiment classification using fine-tuned RoBERTa model")
        with col2:
            st.markdown("### 📊 Topic Modeling")
            st.write("Discover key themes and issues with BERTopic")
        with col3:
            st.markdown("### 🔍 Deep Insights")
            st.write("Version analysis, n-grams, and customer voice exploration")
    
else:
    # Apply global sentiment filter
    if 'sentiment_filter' in locals():
        df = filter_by_sentiment(st.session_state.reviews_df, sentiment_filter)
    else:
        df = st.session_state.reviews_df
    
    if df.empty:
        st.warning("No reviews match the selected filters.")
        st.stop()
    
    # =============================================================================
    # Row 1: Executive Summary (KPIs)
    # =============================================================================
    
    st.subheader("📈 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_reviews = len(df)
        st.metric(
            label="Total Reviews",
            value=f"{total_reviews:,}",
            delta=None
        )
    
    with col2:
        avg_rating = df['rating'].mean()
        st.metric(
            label="Avg Rating",
            value=f"{avg_rating:.2f} ⭐"
        )
    
    with col3:
        sentiment_counts = df['predicted_sentiment'].value_counts()
        pos_pct = (sentiment_counts.get('Positive', 0) / len(df)) * 100
        neu_pct = (sentiment_counts.get('Neutral', 0) / len(df)) * 100
        neg_pct = (sentiment_counts.get('Negative', 0) / len(df)) * 100
        
        if pos_pct >= 50:
            net_sentiment = "Positive"
        elif pos_pct >= 30:
            net_sentiment = "Neutral"
        else:
            net_sentiment = "Negative"
        
        st.metric(
            label="Net Sentiment",
            value=net_sentiment,
            delta=None,
            help=f"Distribusi: Positive {pos_pct:.1f}% | Neutral {neu_pct:.1f}% | Negative {neg_pct:.1f}%"
        )
    
    with col4:
        if 'topic' in df.columns and st.session_state.topic_labels:
            top_topic_num = df['topic'].mode()[0] if not df['topic'].mode().empty else -1
            top_topic = st.session_state.topic_labels.get(top_topic_num, 'Unknown')
        else:
            top_topic = "N/A"
        
        st.metric(
            label="Top Issue",
            value=top_topic[:30] + "..." if len(top_topic) > 30 else top_topic,
            delta=None
        )
    
    st.markdown("---")
    
    # =============================================================================
    # Row 2: Sentiment Deep Dive
    # =============================================================================
    
    st.subheader("💭 Sentiment Deep Dive")
    
    col_left, col_right = st.columns([3, 7])
    
    with col_left:
        # Donut chart for sentiment distribution
        sentiment_counts = df['predicted_sentiment'].value_counts()
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.5,
            marker=dict(colors=['#2ecc71', '#95a5a6', '#e74c3c']),
            textinfo='label+percent',
            textposition='outside',
            pull=[0.05, 0, 0]
        )])
        
        fig_donut.update_layout(
            title="Sentiment Distribution",
            height=350,
            showlegend=True,
            margin=dict(l=50, r=50, t=40, b=40)  # Increased margins to prevent label cutoff
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_right:
        # Timeline trend with dual axis - adaptive frequency
        # Calculate date range to determine optimal aggregation frequency
        date_range = (df['date'].max() - df['date'].min()).days
        
        # Adaptive frequency selection:
        # - Daily: <= 30 days (1 month)
        # - Weekly: 31-180 days (~1-6 months)
        # - Monthly: > 180 days (> 6 months)
        if date_range <= 30:
            freq = 'D'
            freq_label = 'Daily'
        elif date_range <= 180:
            freq = 'W'
            freq_label = 'Weekly'
        else:
            freq = 'M'
            freq_label = 'Monthly'
        
        df_timeline = aggregate_by_date(df, freq=freq)
        
        if not df_timeline.empty:
            fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Sentiment score line
            fig_timeline.add_trace(
                go.Scatter(
                    x=df_timeline['date'],
                    y=df_timeline['avg_sentiment'],
                    name="Avg Sentiment",
                    line=dict(color='#3498db', width=2),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
            
            # Rating line
            fig_timeline.add_trace(
                go.Scatter(
                    x=df_timeline['date'],
                    y=df_timeline['avg_rating'],
                    name="Avg Rating",
                    line=dict(color='#f39c12', width=2),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            fig_timeline.update_xaxes(title_text="Date")
            fig_timeline.update_yaxes(title_text="Sentiment Score (0-2)", secondary_y=False)
            fig_timeline.update_yaxes(title_text="Star Rating (1-5)", secondary_y=True)
            
            fig_timeline.update_layout(
                title=f"Sentiment & Rating Trend Over Time ({freq_label})",
                hovermode='x unified',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Not enough data for timeline visualization")
    
    st.markdown("---")
    
    # =============================================================================
    # Row 3: Topic Modeling & Context
    # =============================================================================
    
    st.subheader("🗂️ Topic Modeling & Customer Voice")
    
    if 'topic' in df.columns and st.session_state.topic_labels:
        # Topic selector
        topic_options = ["All Topics"] + [f"{k}: {v}" for k, v in st.session_state.topic_labels.items() if k != -1]
        selected_topic_display = st.selectbox("Select Topic to Explore:", topic_options)
        
        col_left, col_right = st.columns([6, 4])
        
        with col_left:
            # Topic frequency bar chart
            topic_counts = df['topic'].value_counts().head(10)
            topic_labels_list = [st.session_state.topic_labels.get(t, f"Topic {t}") for t in topic_counts.index]
            
            fig_topics = go.Figure(data=[
                go.Bar(
                    y=topic_labels_list,
                    x=topic_counts.values,
                    orientation='h',
                    marker=dict(
                        color=['#1f77b4' if selected_topic_display == "All Topics" or 
                               str(topic_counts.index[i]) in selected_topic_display 
                               else '#95a5a6' for i in range(len(topic_counts))]
                    )
                )
            ])
            
            fig_topics.update_layout(
                title="Top Topics by Frequency",
                xaxis_title="Number of Reviews",
                yaxis_title="Topic",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_topics, use_container_width=True)
        
        with col_right:
            st.markdown("#### 💬 Representative Reviews")
            
            # Filter by selected topic
            if selected_topic_display != "All Topics":
                topic_num = int(selected_topic_display.split(":")[0])
                topic_df = df[df['topic'] == topic_num].copy()
            else:
                topic_df = df.copy()
            
            # Sort by thumbs_up and get top 3
            top_reviews = topic_df.nlargest(3, 'thumbs_up')
            
            if not top_reviews.empty:
                for idx, row in top_reviews.iterrows():
                    st.markdown(f"""
                    <div class="review-card">
                        <strong>{row['user_name']}</strong> | ⭐ {row['rating']} | 👍 {row['thumbs_up']}<br>
                        <em>"{row['review_text'][:200]}{'...' if len(row['review_text']) > 200 else ''}"</em>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No reviews available for this topic")
    else:
        st.info("Topic modeling data not available")
    
    st.markdown("---")
    
    # =============================================================================
    # Row 4: Technical Insights
    # =============================================================================
    
    st.subheader("🔧 Technical Insights")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 📱 Version Analysis")
        
        # Version sentiment analysis
        version_sentiment = df.groupby(['app_version', 'predicted_sentiment']).size().unstack(fill_value=0)
        
        if not version_sentiment.empty:
            # Get top 10 versions by review count
            top_versions = df['app_version'].value_counts().head(10).index
            version_sentiment_top = version_sentiment.loc[version_sentiment.index.isin(top_versions)]
            
            fig_version = go.Figure()
            
            colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
            for sentiment in ['Negative', 'Neutral', 'Positive']:
                if sentiment in version_sentiment_top.columns:
                    fig_version.add_trace(go.Bar(
                        name=sentiment,
                        x=version_sentiment_top.index,
                        y=version_sentiment_top[sentiment],
                        marker_color=colors.get(sentiment, '#3498db')
                    ))
            
            fig_version.update_layout(
                barmode='stack',
                xaxis_title="App Version",
                yaxis_title="Number of Reviews",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig_version, use_container_width=True)
        else:
            st.info("Version data not available")
    
    with col_right:
        st.markdown("#### 🔍 Negative Review Keywords")
        
        # Extract bigrams from negative reviews
        negative_reviews = df[df['predicted_sentiment'] == 'Negative']['review_text'].tolist()
        
        if negative_reviews:
            ngram_df = extract_ngrams(negative_reviews, n=2, top_k=10)
            
            if not ngram_df.empty:
                fig_ngrams = px.bar(
                    ngram_df,
                    x='frequency',
                    y='ngram',
                    orientation='h',
                    color='frequency',
                    color_continuous_scale='Reds'
                )
                
                fig_ngrams.update_layout(
                    xaxis_title="Frequency",
                    yaxis_title="Bigram",
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig_ngrams, use_container_width=True)
            else:
                st.info("Not enough data for n-gram analysis")
        else:
            st.info("No negative reviews found")
    
    st.markdown("---")
    
    # =============================================================================
    # Row 5: Raw Data Explorer
    # =============================================================================
    
    st.subheader("📋 Raw Data Explorer")
    
    # Prepare display dataframe
    display_df = df[['date', 'app_version', 'rating', 'predicted_sentiment', 'review_text']].copy()
    
    if 'topic' in df.columns:
        display_df['topic_label'] = df['topic'].map(st.session_state.topic_labels)
        display_df = display_df[['date', 'app_version', 'rating', 'predicted_sentiment', 'topic_label', 'review_text']]
    
    # Format date column
    display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Rename columns for display
    display_df.columns = ['Date', 'Version', 'Rating', 'Sentiment', 'Topic', 'Review'] if 'topic_label' in display_df.columns else ['Date', 'Version', 'Rating', 'Sentiment', 'Review']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Data (CSV)",
        data=csv,
        file_name=f"reviews_{st.session_state.selected_app['title']}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )