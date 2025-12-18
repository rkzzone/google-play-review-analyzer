"""
App Review Sentiment & Topic Intelligence Dashboard
Production-ready Streamlit application for analyzing Google Play Store reviews
Memory-optimized for Streamlit Cloud
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
import gc  # Memory management

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils import (
    search_app_hybrid,
    scrape_app_reviews,
    load_sentiment_models,
    predict_sentiment_batch,
    generate_topics,
    get_topic_labels,
    extract_ngrams,
    aggregate_by_date,
    filter_by_sentiment,
    generate_pdf_report
)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Analisis Review Aplikasi Indonesia",
    page_icon="🇮🇩",
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
st.markdown("**Analisis review Google Play Store dengan AI - Sentiment Analysis & Topic Modeling**")

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

# Search Section (Always visible)
st.markdown("Cari aplikasi untuk dianalisis review-nya dengan AI")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Cari Aplikasi",
        placeholder="Nama aplikasi atau Link Play Store",
        label_visibility="collapsed",
        key="app_search_query"
    )
with col2:
    search_clicked = st.button("🔍 Cari", width='stretch', type="primary")

if search_clicked:
    if query:
        with st.spinner("Sedang mencari di Play Store..."):
            results = search_app_hybrid(query)
            
            if results:
                st.session_state.search_results = results
                st.success(f"✅ Ditemukan {len(results)} aplikasi!")
            else:
                st.error("❌ Aplikasi tidak ditemukan. Pastikan ejaan benar atau coba paste Link Play Store-nya.")
    else:
        st.warning("⚠️ Masukkan kata kunci pencarian.")

# Display search results
if 'search_results' in st.session_state and st.session_state.search_results:
    results = st.session_state.search_results[:10]
    
    # If app selected, show only selected app + config
    if st.session_state.selected_app:
        selected_app = st.session_state.selected_app
        
        # Row 1: Selected app (left) + Scraping config (right)
        col_app, col_config = st.columns([1, 2])
        
        with col_app:
            with st.container(border=True):
                c_img, c_txt = st.columns([1, 2])
                with c_img:
                    if selected_app.get('icon'):
                        st.image(selected_app['icon'], width='stretch')
                    else:
                        st.markdown("<div style='text-align: center; font-size: 40px;'>📱</div>", unsafe_allow_html=True)
                with c_txt:
                    st.markdown(f"**{selected_app['title']}**")
                    score = selected_app.get('score', 0)
                    st.caption(f"⭐ {score:.2f}" if score else "⭐ N/A")
                    st.caption(f"👨‍💻 {selected_app.get('developer', 'Unknown')}")
        
        with col_config:
            with st.container(border=True):
                filter_mode = st.radio(
                    "Scraping Mode:",
                    options=["Review Count", "Date Range"],
                    horizontal=True
                )
                
                if filter_mode == "Review Count":
                    review_count = st.number_input(
                        "Number of Reviews",
                        min_value=50,
                        max_value=5000,
                        value=500,
                        step=50
                    )
                    start_date = None
                    end_date = None
                else:
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
        
        # Row 2: Start Analysis Button
        if st.button("🚀 Start Analysis", width='stretch', type="primary"):
            app_id = selected_app['appId']
            
            # Scrape reviews
            if filter_mode == "Review Count":
                reviews_df = scrape_app_reviews(
                    app_id=app_id,
                    lang='id',
                    country='id',
                    filter_mode='count',
                    target_count=review_count
                )
            else:
                reviews_df = scrape_app_reviews(
                    app_id=app_id,
                    lang='id',
                    country='id',
                    filter_mode='date_range',
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date)
                )
            
            if not reviews_df.empty:
                # Single status container for model loading
                status_container = st.empty()
                
                # Load Indonesian sentiment model only
                status_container.info("📥 Loading Indonesian sentiment model from HuggingFace...")
                models_dict = load_sentiment_models(load_mode='id')
                
                if models_dict and models_dict.get('id'):
                    status_container.success("✅ Indonesian model loaded!")
                    
                    # Predict sentiment
                    with st.spinner("🤖 Analyzing sentiment..."):
                        predictions, probabilities, detected_langs = predict_sentiment_batch(
                            reviews_df['review_text'].tolist(),
                            models_dict,
                            language_mode='id'
                        )
                        reviews_df['predicted_sentiment'] = predictions
                        reviews_df['detected_language'] = detected_langs
                        
                        # Clear GPU memory after sentiment analysis
                        gc.collect()
                    
                    # Generate topics
                    with st.spinner("📊 Discovering topics..."):
                        topics, topic_model, topic_info = generate_topics(
                            reviews_df['review_text'].tolist(),
                            min_topic_size=max(5, len(reviews_df) // 20)
                        )
                        
                        if topics is not None and topic_model is not None:
                            reviews_df['topic'] = topics
                            st.session_state.topic_model = topic_model
                            st.session_state.topic_labels = get_topic_labels(topic_model, topic_info)
                            st.success("✅ Topic modeling complete!")
                        else:
                            st.warning("⚠️ Topic modeling skipped or failed. Continuing with sentiment analysis only...")
                            reviews_df['topic'] = -1
                            st.session_state.topic_model = None
                            st.session_state.topic_labels = {}
                        
                        # Clear memory after topic modeling
                        gc.collect()
                    
                    # Store in session state
                    st.session_state.reviews_df = reviews_df
                    st.success("✅ Analysis complete!")
                    st.rerun()
            else:
                st.error("❌ Failed to fetch reviews. Please try again.")
    
    else:
        # Show all search results (app not yet selected)
        apps_per_row = 5
        
        for i in range(0, len(results), apps_per_row):
            row_apps = results[i:i+apps_per_row]
            cols = st.columns(apps_per_row)
            
            for idx, app in enumerate(row_apps):
                with cols[idx]:
                    with st.container(border=True):
                        c_img, c_txt = st.columns([1, 3])
                        
                        with c_img:
                            if app.get('icon'):
                                st.image(app['icon'], width='stretch')
                            else:
                                st.markdown("<div style='text-align: center; font-size: 40px;'>📱</div>", unsafe_allow_html=True)
                        
                        with c_txt:
                            title = app['title']
                            max_title_len = 15
                            display_title = title[:max_title_len] + "..." if len(title) > max_title_len else title
                            st.markdown(f"**{display_title}**", help=title)
                            
                            score = app.get('score', 0)
                            rating_display = f"{score:.2f}" if score else "N/A"
                            dev = app.get('developer', 'Unknown')
                            max_dev_len = 8
                            display_dev = dev[:max_dev_len] + "..." if len(dev) > max_dev_len else dev
                            st.caption(f"⭐ {rating_display} | 👨‍💻 {display_dev}")

                        unique_key = f"btn_{app['appId']}_{i}_{idx}"
                        if st.button("Pilih", key=unique_key, width='stretch', type="secondary"):
                            st.session_state.selected_app = app
                            st.rerun()

# Show analysis results below if available
if st.session_state.reviews_df is not None:
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
        
        st.plotly_chart(fig_donut, use_container='stretch')
    
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
            
            st.plotly_chart(fig_timeline, use_container='stretch')
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
            # Topic frequency bar chart (sorted descending, top to bottom)
            topic_counts = df['topic'].value_counts().head(10)
            topic_labels_list = [st.session_state.topic_labels.get(t, f"Topic {t}") for t in topic_counts.index]
            
            # Reverse order untuk descending dari atas ke bawah
            topic_labels_list_reversed = topic_labels_list[::-1]
            topic_counts_reversed = topic_counts.values[::-1]
            topic_index_reversed = topic_counts.index[::-1]
            
            fig_topics = go.Figure(data=[
                go.Bar(
                    y=topic_labels_list_reversed,
                    x=topic_counts_reversed,
                    orientation='h',
                    marker=dict(
                        color=['#1f77b4' if selected_topic_display == "All Topics" or 
                               str(topic_index_reversed[i]) in selected_topic_display 
                               else '#95a5a6' for i in range(len(topic_counts_reversed))]
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
            
            st.plotly_chart(fig_topics, use_container='stretch')
        
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
        st.info("💡 Topic modeling data not available. This can happen when:")
        st.markdown("""
        - Reviews are too few (need at least 10-20 reviews)
        - Reviews are too similar to each other
        - Text preprocessing removed too much content
        
        **Try:**
        - Scraping more reviews (increase count or date range)
        - Using a different app with more diverse reviews
        """)
    
    st.markdown("---")
    
    # =============================================================================
    # Row 4: Technical Insights
    # =============================================================================
    
    st.subheader("🔧 Technical Insights")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 📱 Version Analysis")
        
        # Toggle untuk filter unknown versions
        exclude_unknown = st.checkbox(
            "Exclude unknown versions",
            value=True,
            help="Filter out 'Unknown', 'Varies with device', and similar generic version strings"
        )
        
        # Version sentiment analysis
        df_version = df.copy()
        
        # Filter unknown versions jika checkbox aktif
        if exclude_unknown:
            # Pattern untuk unknown versions (case insensitive)
            unknown_patterns = ['unknown', 'varies with device', 'varies', 'variable', 'n/a', 'null', 'none']
            
            # Filter dengan handling NaN dan case insensitive
            for pattern in unknown_patterns:
                df_version = df_version[
                    ~df_version['app_version'].astype(str).str.lower().str.contains(pattern, na=False)
                ]
            
            # Filter version yang kosong atau hanya angka 0
            df_version = df_version[
                (df_version['app_version'].astype(str).str.strip() != '') & 
                (df_version['app_version'].astype(str) != '0')
            ]
        
        if not df_version.empty and len(df_version) > 0:
            version_sentiment = df_version.groupby(['app_version', 'predicted_sentiment']).size().unstack(fill_value=0)
            
            if not version_sentiment.empty:
                # Get top 10 versions by review count
                top_versions = df_version['app_version'].value_counts().head(10).index
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
                    height=300,
                    margin=dict(l=20, r=20, t=10, b=20)
                )
                
                st.plotly_chart(fig_version, use_container='stretch')
            else:
                st.info("No version data available")
        else:
            st.warning(f"All {len(df)} reviews filtered out. Try unchecking 'Exclude unknown versions'.")
    
    with col_right:
        st.markdown("#### 🔍 Negative Review Keywords")
        
        # Extract bigrams from negative reviews
        negative_reviews = df[df['predicted_sentiment'] == 'Negative']['review_text'].tolist()
        
        if negative_reviews:
            ngram_df = extract_ngrams(negative_reviews, n=2, top_k=10)
            
            if not ngram_df.empty:
                # Reverse order untuk descending dari atas ke bawah
                ngram_df_reversed = ngram_df.iloc[::-1].copy()
                
                fig_ngrams = px.bar(
                    ngram_df_reversed,
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
                
                st.plotly_chart(fig_ngrams, use_container='stretch')
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
        width='stretch',
        height=400
    )
    
    # Download PDF Report Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📄 Download Professional Report (PDF)", width='stretch', type="primary"):
            with st.spinner("Generating professional report..."):
                pdf_buffer = generate_pdf_report(df, st.session_state.selected_app, st.session_state.topic_labels)
                
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"Review_Analysis_Report_{st.session_state.selected_app['title'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    width='stretch'
                )