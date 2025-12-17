"""
Utility functions for App Review Sentiment Analysis Dashboard
"""

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from google_play_scraper import app, search, Sort, reviews
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from rapidfuzz import fuzz
import re

# Base path configuration - use script directory
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'saved_model')


# =============================================================================
# Google Play Scraper Functions
# =============================================================================
import re
import time
from google_play_scraper import search, app as app_detail
from rapidfuzz import fuzz
# Pastikan library ini terinstall: pip install googlesearch-python
from googlesearch import search as google_web_search

def parse_installs(installs_str):
    if not installs_str: return 0
    clean_str = re.sub(r'[^\d]', '', str(installs_str))
    try: return int(clean_str)
    except ValueError: return 0

def get_id_from_web_search(query):
    """
    Fallback: Mencari ID lewat Google.com dengan strategi ganda.
    """
    print(f"🌍 Memulai Web Search Fallback untuk: {query}")
    
    # STRATEGI 1: Query Spesifik (site:play.google.com)
    # Ini paling akurat tapi kadang hasil sedikit
    queries = [
        f"site:play.google.com/store/apps/details {query}",
        f"{query} google play store app" # Strategi 2: Query Natural (Backup)
    ]
    
    for q in queries:
        try:
            # Ambil 3 hasil teratas, pause 1 detik biar ga dikira bot
            results = list(google_web_search(q, num_results=3, lang='id', sleep_interval=1))
            
            for url in results:
                # Coba ekstrak ID dari URL
                # Pola regex menangkap id=com.xxx.yyy
                id_pattern = r'id=([a-zA-Z0-9\._]+)'
                match = re.search(id_pattern, url)
                
                if match:
                    found_id = match.group(1)
                    # Filter ID sampah (kadang ada id=googlevideo dll)
                    if "." in found_id: 
                        print(f"✅ Web Search menemukan ID: {found_id} dari query '{q}'")
                        return found_id
                        
        except Exception as e:
            print(f"⚠️ Web search error pada query '{q}': {e}")
            continue
            
    print("❌ Web Search gagal menemukan ID.")
    return None

def search_app_hybrid(query, lang='id', country='id'):
    """
    Hybrid Search: Link/ID -> Play Store Search -> Google Web Search Fallback
    """
    formatted_results = []
    
    # --- LAPIS 1: DETEKSI INPUT LINK / ID (BYPASS) ---
    id_pattern = r'id=([a-zA-Z0-9\._]+)'
    potential_id = None
    link_match = re.search(id_pattern, query)
    
    if link_match:
        potential_id = link_match.group(1)
    elif "." in query and " " not in query:
        potential_id = query.strip()
        
    if potential_id:
        try:
            r = app_detail(potential_id, lang=lang, country=country)
            return [{
                'appId': r['appId'],
                'title': r['title'],
                'icon': r['icon'],
                'score': r['score'],
                'developer': r['developer'],
                'source': 'Direct Link'
            }]
        except:
            pass 

    # --- LAPIS 2: PLAY STORE SEARCH (INTERNAL) ---
    exact_match_found = False
    
    try:
        results = search(query, lang=lang, country=country, n_hits=30)
        search_term = query.lower().strip()
        
        if results:
            for r in results:
                # Validasi ID: Skip jika ID None
                if not r.get('appId'): 
                    continue
                
                title_lower = r['title'].lower()
                
                # Cek Exact Match
                if title_lower == search_term:
                    exact_match_found = True

                # Ranking Logic
                installs_count = parse_installs(r.get('installs', '0'))
                score = r.get('score', 0) if r.get('score') else 0
                
                relevance = 0
                relevance += fuzz.partial_ratio(search_term, title_lower)
                if title_lower.startswith(search_term): relevance += 50
                if installs_count > 100_000_000: relevance += 200
                elif installs_count > 10_000_000: relevance += 150
                relevance += (score * 5)
                
                formatted_results.append({
                    'appId': r['appId'],
                    'title': r['title'],
                    'icon': r.get('icon', ''),
                    'score': score,
                    'developer': r.get('developer', ''),
                    'relevance': relevance
                })
    except Exception as e:
        print(f"Internal search error: {e}")

    # --- LAPIS 3: GOOGLE WEB SEARCH (FALLBACK) ---
    # Jalan jika: Hasil kosong ATAU Tidak ada yang namanya SAMA PERSIS (Exact Match)
    should_run_fallback = len(formatted_results) == 0 or not exact_match_found
    
    if should_run_fallback:
        print(f"⚡ Mengaktifkan Fallback... (Exact Match: {exact_match_found})")
        fallback_id = get_id_from_web_search(query)
        
        if fallback_id:
            # Cek duplikasi agar tidak double
            existing_ids = [app['appId'] for app in formatted_results]
            
            if fallback_id not in existing_ids:
                try:
                    # Ambil detail dari ID yang ditemukan Web Search
                    r = app_detail(fallback_id, lang=lang, country=country)
                    
                    # Masukkan ke urutan PERTAMA (Top 1)
                    formatted_results.insert(0, {
                        'appId': r['appId'],
                        'title': r['title'],
                        'icon': r['icon'],
                        'score': r['score'],
                        'developer': r['developer'],
                        'relevance': 9999 # Paksa skor tertinggi
                    })
                except Exception as e:
                    print(f"Gagal mengambil detail fallback ID: {e}")

    # Sorting Final
    formatted_results.sort(key=lambda x: x['relevance'], reverse=True)
    return formatted_results[:12]

def scrape_app_reviews(app_id, lang='en', country='us', filter_mode='count', 
                       target_count=500, start_date=None, end_date=None):
    """
    Scrape app reviews from Google Play Store with multiple filter modes.
    
    Args:
        app_id (str): Google Play app ID
        lang (str): Language code
        country (str): Country code
        filter_mode (str): 'count' or 'date_range'
        target_count (int): Number of reviews to fetch (for count mode)
        start_date (datetime): Start date for filtering (for date_range mode)
        end_date (datetime): End date for filtering (for date_range mode)
        
    Returns:
        pd.DataFrame: DataFrame with review data
    """
    all_reviews = []
    continuation_token = None
    
    try:
        if filter_mode == 'count':
            # Fetch specific number of reviews
            batch_size = min(200, target_count)
            
            with st.spinner(f'Fetching {target_count} reviews...'):
                progress_bar = st.progress(0)
                
                while len(all_reviews) < target_count:
                    result, continuation_token = reviews(
                        app_id,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=batch_size,
                        continuation_token=continuation_token
                    )
                    
                    if not result:
                        break
                    
                    all_reviews.extend(result)
                    progress_bar.progress(min(len(all_reviews) / target_count, 1.0))
                    
                    if len(all_reviews) >= target_count:
                        all_reviews = all_reviews[:target_count]
                        break
                    
                    if not continuation_token:
                        break
                    
                    time.sleep(1)  # Rate limiting
                
                progress_bar.empty()
        
        elif filter_mode == 'date_range':
            # Fetch reviews within date range
            if not start_date or not end_date:
                st.error("Start and end dates are required for date range mode")
                return pd.DataFrame()
            
            # Convert dates to datetime if needed
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            with st.spinner(f'Fetching reviews from {start_date.date()} to {end_date.date()}...'):
                progress_placeholder = st.empty()
                
                batch_count = 0
                while True:
                    result, continuation_token = reviews(
                        app_id,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=200,
                        continuation_token=continuation_token
                    )
                    
                    if not result:
                        break
                    
                    batch_count += 1
                    progress_placeholder.text(f'Fetched {batch_count * 200} reviews (filtering by date)...')
                    
                    # Check dates in this batch
                    oldest_in_batch = min([r['at'] for r in result])
                    
                    # Add reviews that fall within the date range
                    for review in result:
                        review_date = review['at']
                        if start_date <= review_date <= end_date:
                            all_reviews.append(review)
                    
                    # Stop if oldest review in batch is before start date
                    if oldest_in_batch < start_date:
                        break
                    
                    if not continuation_token:
                        break
                    
                    time.sleep(1)  # Rate limiting
                
                progress_placeholder.empty()
        
        # Convert to DataFrame
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            
            # Select and rename columns
            columns_to_keep = {
                'content': 'review_text',
                'score': 'rating',
                'thumbsUpCount': 'thumbs_up',
                'reviewCreatedVersion': 'app_version',
                'at': 'date',
                'replyContent': 'developer_reply',
                'userName': 'user_name'
            }
            
            df = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
            
            # Clean data
            df['review_text'] = df['review_text'].fillna('')
            df['app_version'] = df['app_version'].fillna('Unknown')
            df['developer_reply'] = df['developer_reply'].fillna('')
            
            # Remove empty reviews
            df = df[df['review_text'].str.strip() != '']
            
            return df.reset_index(drop=True)
        else:
            st.warning("No reviews found matching the criteria")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error scraping reviews: {str(e)}")
        return pd.DataFrame()


# =============================================================================
# Sentiment Analysis Functions
# =============================================================================

@st.cache_resource
def load_sentiment_model():
    """
    Load the fine-tuned RoBERTa model and tokenizer.
    Uses caching to load only once.
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
            return None, None, None
        
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        return None, None, None


def predict_sentiment_batch(texts, model, tokenizer, device, batch_size=32):
    """
    Predict sentiment for a batch of texts.
    
    Args:
        texts (list): List of review texts
        model: Fine-tuned RoBERTa model
        tokenizer: RoBERTa tokenizer
        device: torch device
        batch_size (int): Batch size for inference
        
    Returns:
        tuple: (predictions, probabilities)
    """
    all_predictions = []
    all_probabilities = []
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            # Convert to CPU and numpy
            predictions = predictions.cpu().numpy()
            probs = probs.cpu().numpy()
            
            # Map to labels
            batch_labels = [label_map[p] for p in predictions]
            
            all_predictions.extend(batch_labels)
            all_probabilities.extend(probs)
        
        return all_predictions, all_probabilities
    except Exception as e:
        st.error(f"Error in sentiment prediction: {str(e)}")
        return [], []


# =============================================================================
# Topic Modeling Functions
# =============================================================================

def preprocess_for_topics(texts):
    """
    Preprocess texts for topic modeling.
    
    Args:
        texts (list): List of review texts
        
    Returns:
        list: Cleaned texts
    """
    cleaned = []
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        cleaned.append(text)
    return cleaned


@st.cache_data
def generate_topics(texts, n_topics=10, min_topic_size=10):
    """
    Generate topics from review texts using BERTopic.
    
    Args:
        texts (list): List of review texts
        n_topics (int): Target number of topics
        min_topic_size (int): Minimum topic size
        
    Returns:
        tuple: (topics, topic_model, doc_topics)
    """
    try:
        if len(texts) < min_topic_size:
            st.warning("Not enough reviews for topic modeling")
            return None, None, None
        
        # Preprocess texts
        cleaned_texts = preprocess_for_topics(texts)
        
        # Custom vectorizer to remove common words
        vectorizer_model = CountVectorizer(
            stop_words='english',
            min_df=2,
            ngram_range=(1, 2)
        )
        
        # Initialize BERTopic with lightweight model
        topic_model = BERTopic(
            embedding_model='all-MiniLM-L6-v2',
            vectorizer_model=vectorizer_model,
            nr_topics=n_topics,
            min_topic_size=min_topic_size,
            calculate_probabilities=False,
            verbose=False
        )
        
        # Fit model
        with st.spinner('Analyzing topics...'):
            topics, _ = topic_model.fit_transform(cleaned_texts)
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        
        return topics, topic_model, topic_info
    except Exception as e:
        st.error(f"Error in topic modeling: {str(e)}")
        return None, None, None


def get_topic_labels(topic_model, topic_info):
    """
    Generate readable topic labels from keywords.
    
    Args:
        topic_model: Fitted BERTopic model
        topic_info: Topic information DataFrame
        
    Returns:
        dict: Mapping of topic numbers to labels
    """
    topic_labels = {}
    
    for idx, row in topic_info.iterrows():
        topic_num = row['Topic']
        if topic_num == -1:
            topic_labels[topic_num] = 'Outliers'
        else:
            # Get top 3 keywords
            topic_words = topic_model.get_topic(topic_num)
            if topic_words:
                keywords = [word for word, _ in topic_words[:3]]
                label = ', '.join(keywords).title()
                topic_labels[topic_num] = label
            else:
                topic_labels[topic_num] = f'Topic {topic_num}'
    
    return topic_labels


# =============================================================================
# N-gram Analysis Functions
# =============================================================================

def extract_ngrams(texts, n=2, top_k=10):
    """
    Extract most common n-grams from texts.
    
    Args:
        texts (list): List of texts
        n (int): N-gram size (2 for bigrams)
        top_k (int): Number of top n-grams to return
        
    Returns:
        pd.DataFrame: DataFrame with n-grams and their frequencies
    """
    try:
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            stop_words='english',
            max_features=top_k * 2
        )
        
        X = vectorizer.fit_transform(texts)
        features = vectorizer.get_feature_names_out()
        frequencies = X.sum(axis=0).A1
        
        # Create DataFrame
        ngram_df = pd.DataFrame({
            'ngram': features,
            'frequency': frequencies
        }).sort_values('frequency', ascending=False).head(top_k)
        
        return ngram_df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error extracting n-grams: {str(e)}")
        return pd.DataFrame()


# =============================================================================
# Data Processing Functions
# =============================================================================

def calculate_sentiment_score(sentiment_label):
    """
    Convert sentiment label to numeric score.
    
    Args:
        sentiment_label (str): 'Positive', 'Neutral', or 'Negative'
        
    Returns:
        int: Numeric score (0, 1, or 2)
    """
    score_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    return score_map.get(sentiment_label, 1)


def aggregate_by_date(df, freq='D'):
    """
    Aggregate reviews by date.
    
    Args:
        df (pd.DataFrame): Reviews DataFrame
        freq (str): Frequency for aggregation ('D' for day, 'W' for week)
        
    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    try:
        df['sentiment_score'] = df['predicted_sentiment'].apply(calculate_sentiment_score)
        
        # Group by date
        grouped = df.groupby(pd.Grouper(key='date', freq=freq)).agg({
            'sentiment_score': 'mean',
            'rating': 'mean',
            'review_text': 'count'
        }).reset_index()
        
        grouped.columns = ['date', 'avg_sentiment', 'avg_rating', 'review_count']
        
        return grouped
    except Exception as e:
        st.error(f"Error aggregating by date: {str(e)}")
        return pd.DataFrame()


def filter_by_sentiment(df, selected_sentiments):
    """
    Filter DataFrame by selected sentiments.
    
    Args:
        df (pd.DataFrame): Reviews DataFrame
        selected_sentiments (list): List of sentiment labels to keep
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if not selected_sentiments or len(selected_sentiments) == 3:
        return df
    
    return df[df['predicted_sentiment'].isin(selected_sentiments)]