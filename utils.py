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
import re

# Base path configuration - use script directory
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'saved_model')


# =============================================================================
# Google Play Scraper Functions
# =============================================================================

def search_app_by_name(app_name, lang='en', country='us'):
    """
    Search for apps by name with intelligent fuzzy matching.
    Uses Google Play Store's own search API.
    
    Args:
        app_name (str): Name of the app to search
        lang (str): Language code
        country (str): Country code
        
    Returns:
        list: List of app results ranked by relevance
    """
    try:
        # Search with higher limit to get better results
        results = search(
            app_name,
            lang=id,
            country=id,
            n_hits=8  # Get many results, we'll filter intelligently
        )
        
        if not results:
            return []
        
        formatted_results = []
        
        for r in results:
            try:
                # Validate essential fields
                if not r or not isinstance(r.get('title'), str):
                    continue
                
                app_id = r.get('appId')
                
                # Skip if no appId - these are broken results
                if not app_id:
                    continue
                
                formatted_results.append({
                    'appId': app_id,
                    'title': r['title'],
                    'icon': r.get('icon', ''),
                    'score': r.get('score', 0),
                    'developer': r.get('developer', ''),
                    'relevance': 0  # Will be calculated
                })
                
            except (KeyError, TypeError, AttributeError):
                continue
        
        if not formatted_results:
            return []
        
        # Intelligent ranking using fuzzy string matching
        search_term = app_name.lower().strip()
        
        for app in formatted_results:
            title_lower = app['title'].lower()
            relevance = 0
            
            # 1. Exact match (1000 points)
            if search_term == title_lower:
                relevance = 1000
            
            # 2. Exact word in title (800 points)
            elif f" {search_term} " in f" {title_lower} ":
                relevance = 800
            
            # 3. Title starts with search (700 points)
            elif title_lower.startswith(search_term):
                relevance = 700
            
            # 4. Title ends with search (600 points)
            elif title_lower.endswith(search_term):
                relevance = 600
            
            # 5. Search term contained in title (500 points)
            elif search_term in title_lower:
                relevance = 500
            
            # 6. Fuzzy matching - word overlap
            else:
                search_words = set(search_term.split())
                title_words = set(title_lower.split())
                
                # Common words
                common = search_words & title_words
                if common:
                    relevance = 300 + (len(common) * 50)
                
                # Partial word matches (typo tolerance)
                for s_word in search_words:
                    if len(s_word) >= 3:
                        for t_word in title_words:
                            # Check if words are similar (share 80%+ characters)
                            if t_word.startswith(s_word[:min(len(s_word), len(t_word))-1]):
                                relevance = max(relevance, 250)
            
            # Boost by app quality (rating 0-5 × 20 = 0-100 points)
            relevance += int(app['score'] * 20)
            
            app['relevance'] = relevance
        
        # Sort by relevance (descending)
        formatted_results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Clean up and return top 12
        for app in formatted_results:
            del app['relevance']
        
        return formatted_results[:12]
        
    except Exception as e:
        import streamlit as st
        st.error(f"Search error: {str(e)}")
        return []


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