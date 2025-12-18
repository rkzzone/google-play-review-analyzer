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
import requests
from urllib.parse import quote
from google_play_scraper import search, app as app_detail
from rapidfuzz import fuzz

def parse_installs(installs_str):
    """Mengubah '1,000,000+' jadi integer"""
    if not installs_str: return 0
    clean_str = re.sub(r'[^\d]', '', str(installs_str))
    try: return int(clean_str)
    except ValueError: return 0

def find_real_app_id_scraping(app_name, developer=None):
    """
    Fungsi Penyelamat (Rescue Function):
    Mencari App ID langsung dari HTML halaman pencarian Google Play
    ketika API mengembalikan None.
    """
    try:
        # Method 1: Scrape HTML Google Play Search
        encoded_name = quote(app_name)
        search_url = f"https://play.google.com/store/search?q={encoded_name}&c=apps"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Request langsung ke Google Play
        response = requests.get(search_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            # Regex untuk mencari pola id=com.paket.nama
            pattern = r'/store/apps/details\?id=([a-zA-Z0-9._]+)'
            matches = re.findall(pattern, response.text)
            
            if matches:
                # Filter hasil sampah (kadang ada id aneh), pastikan ada titik
                for candidate_id in matches:
                    if "." in candidate_id:
                        print(f"✅ ID ditemukan via Scraping HTML: {candidate_id}")
                        return candidate_id
        
        # Method 2: Fallback - Coba search ulang dengan region en/us + developer
        if developer:
            print(f"🔄 Fallback: Mencoba search dengan developer di region EN/US...")
            try:
                alt_results = search(f"{app_name} {developer}", lang='en', country='us', n_hits=5)
                for r in alt_results:
                    alt_id = r.get('appId')
                    if alt_id and alt_id != 'None' and alt_id != '':
                        print(f"✅ ID ditemukan via Fallback Search: {alt_id}")
                        return alt_id
            except:
                pass
        
        return None
    except Exception as e:
        print(f"⚠️ Error scraping HTML: {e}")
        return None

def search_app_hybrid(query, lang='id', country='id'):
    """
    Hybrid Search v2: Link/ID -> Internal Search -> HTML Scraping Repair
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
                'relevance': 9999 # Prioritas Tertinggi
            }]
        except:
            pass 

    # --- LAPIS 2: INTERNAL SEARCH + AUTO REPAIR ---
    try:
        # Kita ambil hasil search biasa
        results = search(query, lang=lang, country=country, n_hits=30)
        search_term = query.lower().strip()
        
        if results:
            for r in results:
                app_id = r.get('appId')
                title = r.get('title', 'Unknown')
                
                # --- LOGIKA PERBAIKAN (YOUR CODE LOGIC) ---
                # Jika ID None, jangan dibuang! Coba kita perbaiki (Scraping)
                if not app_id or app_id == 'None' or app_id == '':
                    print(f"🔍 ID Kosong untuk '{title}', mencoba memperbaiki...")
                    developer_name = r.get('developer')
                    recovered_id = find_real_app_id_scraping(title, developer_name)
                    
                    if recovered_id:
                        # Jika berhasil diperbaiki, kita harus ambil detailnya lagi
                        # karena data di 'r' saat ini mungkin tidak lengkap/rusak
                        try:
                            fixed_app = app_detail(recovered_id, lang=lang, country=country)
                            # Update data 'r' dengan data yang baru ditarik
                            r = fixed_app
                            app_id = recovered_id # Update variabel lokal
                        except:
                            continue # Jika fetch detail gagal, ya sudah skip
                    else:
                        continue # Jika tidak bisa diperbaiki, skip
                
                # --- PROSES RANKING ---
                title_lower = r['title'].lower()
                installs_count = parse_installs(r.get('installs', '0'))
                score = r.get('score', 0) if r.get('score') else 0
                
                relevance = 0
                relevance += fuzz.partial_ratio(search_term, title_lower)
                
                if title_lower == search_term: relevance += 100
                elif title_lower.startswith(search_term): relevance += 50
                
                if installs_count > 100_000_000: relevance += 200
                elif installs_count > 10_000_000: relevance += 150
                
                relevance += (score * 5)
                
                formatted_results.append({
                    'appId': app_id, # Pastikan pakai ID yang valid (atau hasil repair)
                    'title': r['title'],
                    'icon': r.get('icon', ''),
                    'score': score,
                    'developer': r.get('developer', ''),
                    'relevance': relevance
                })
                
    except Exception as e:
        print(f"Search error: {e}")

    # Sorting Final
    formatted_results.sort(key=lambda x: x['relevance'], reverse=True)
    return formatted_results[:10]

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
    
    Tries to load from local path first, then falls back to Hugging Face Hub.
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    try:
        # Try local path first (for local development)
        if os.path.exists(MODEL_PATH):
            st.info("Loading model from local path...")
            tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
            model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        else:
            # Fallback: Load from Hugging Face Hub (for Streamlit Cloud)
            st.info("Local model not found. Loading from Hugging Face Hub...")
            model_name = "rkkzone/roberta-sentiment-playstore"  # Your fine-tuned model on HF
            
            try:
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                model = RobertaForSequenceClassification.from_pretrained(model_name)
                st.success("✅ Fine-tuned model loaded from Hugging Face Hub!")
            except Exception as hf_error:
                # If HF model not found, use base model as fallback
                st.warning(f"Hugging Face model not found. Using base roberta-base model instead.")
                st.warning("⚠️ This model is NOT fine-tuned for sentiment analysis!")
                tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
        
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


# =============================================================================
# PDF Report Generation
# =============================================================================

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime


def generate_pdf_report(df, app_info, topic_labels):
    """
    Generate professional consultant-style PDF report in 16:9 format.
    
    Args:
        df (pd.DataFrame): Reviews DataFrame with analysis results
        app_info (dict): App information
        topic_labels (dict): Topic labels mapping
        
    Returns:
        BytesIO: PDF file buffer
    """
    buffer = BytesIO()
    
    # 16:9 landscape format (presentation style)
    page_width = 11 * inch
    page_height = 6.1875 * inch  # 16:9 ratio
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=(page_width, page_height),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#555555'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14
    )
    
    # Story elements
    story = []
    
    # === PAGE 1: Cover Page ===
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("App Review Intelligence Report", title_style))
    story.append(Paragraph(f"<b>{app_info['title']}</b>", subtitle_style))
    story.append(Paragraph(f"by {app_info.get('developer', 'Unknown')}", body_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}", body_style))
    story.append(Paragraph(f"Total Reviews Analyzed: {len(df):,}", body_style))
    story.append(PageBreak())
    
    # === PAGE 2: Executive Summary ===
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Calculate metrics
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    sentiment_counts = df['predicted_sentiment'].value_counts()
    pos_pct = (sentiment_counts.get('Positive', 0) / len(df)) * 100
    neu_pct = (sentiment_counts.get('Neutral', 0) / len(df)) * 100
    neg_pct = (sentiment_counts.get('Negative', 0) / len(df)) * 100
    
    # Summary table
    summary_data = [
        ['Metric', 'Value', 'Insight'],
        ['Total Reviews', f'{total_reviews:,}', f'Sample size: {"Excellent" if total_reviews > 1000 else "Good" if total_reviews > 500 else "Moderate"}'],
        ['Average Rating', f'{avg_rating:.2f} / 5.0', f'{"Strong" if avg_rating >= 4.0 else "Good" if avg_rating >= 3.5 else "Needs improvement"}'],
        ['Sentiment Distribution', 
         f'Pos: {pos_pct:.1f}% | Neu: {neu_pct:.1f}% | Neg: {neg_pct:.1f}%',
         f'Overall: {"Positive" if pos_pct >= 50 else "Neutral" if pos_pct >= 30 else "Negative"}']
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Key Insights
    story.append(Paragraph("<b>Key Insights:</b>", body_style))
    insights = []
    
    if avg_rating >= 4.0:
        insights.append(f"\u2022 Users are highly satisfied with an average rating of {avg_rating:.2f}/5.0")
    elif avg_rating >= 3.5:
        insights.append(f"\u2022 Overall satisfaction is good ({avg_rating:.2f}/5.0) with room for improvement")
    else:
        insights.append(f"\u2022 Rating of {avg_rating:.2f}/5.0 indicates significant user concerns requiring attention")
    
    if pos_pct >= 60:
        insights.append(f"\u2022 Strong positive sentiment ({pos_pct:.1f}%) indicates excellent user experience")
    elif neg_pct >= 30:
        insights.append(f"\u2022 High negative sentiment ({neg_pct:.1f}%) requires immediate action")
    
    if neu_pct >= 30:
        insights.append(f"\u2022 Significant neutral sentiment ({neu_pct:.1f}%) suggests opportunity to convert users to advocates")
    
    for insight in insights:
        story.append(Paragraph(insight, body_style))
    
    story.append(PageBreak())
    
    # === PAGE 3: Sentiment Analysis Detail ===
    story.append(Paragraph("Sentiment Analysis Breakdown", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    sentiment_data = [
        ['Sentiment', 'Count', 'Percentage', 'Recommendation'],
        ['Positive', 
         f'{sentiment_counts.get("Positive", 0):,}', 
         f'{pos_pct:.1f}%',
         'Leverage these reviews for marketing'],
        ['Neutral', 
         f'{sentiment_counts.get("Neutral", 0):,}', 
         f'{neu_pct:.1f}%',
         'Identify features to improve engagement'],
        ['Negative', 
         f'{sentiment_counts.get("Negative", 0):,}', 
         f'{neg_pct:.1f}%',
         'Priority: Address critical issues']
    ]
    
    sentiment_table = Table(sentiment_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 4.5*inch])
    sentiment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda')),
        ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#fff3cd')),
        ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#f8d7da'))
    ]))
    
    story.append(sentiment_table)
    story.append(PageBreak())
    
    # === PAGE 4: Top Topics ===
    if 'topic' in df.columns and topic_labels:
        story.append(Paragraph("Top Discussion Topics", heading_style))
        story.append(Spacer(1, 0.2*inch))
        
        topic_counts = df['topic'].value_counts().head(8)
        topic_data = [['Rank', 'Topic', 'Mentions', '% of Reviews']]
        
        for rank, (topic_num, count) in enumerate(topic_counts.items(), 1):
            topic_name = topic_labels.get(topic_num, f'Topic {topic_num}')
            pct = (count / len(df)) * 100
            topic_data.append([
                str(rank),
                topic_name[:60] + '...' if len(topic_name) > 60 else topic_name,
                f'{count:,}',
                f'{pct:.1f}%'
            ])
        
        topic_table = Table(topic_data, colWidths=[0.7*inch, 6*inch, 1.3*inch, 1*inch])
        topic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))
        
        story.append(topic_table)
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<b>Strategic Recommendation:</b> Focus product development on top 3 topics which represent majority of user concerns.", body_style))
        story.append(PageBreak())
    
    # === PAGE 5: Recommendations ===
    story.append(Paragraph("Strategic Recommendations", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    recommendations = []
    
    if neg_pct >= 20:
        negative_reviews = df[df['predicted_sentiment'] == 'Negative']
        if len(negative_reviews) > 0:
            recommendations.append(f"<b>1. Address Critical Issues ({neg_pct:.1f}% negative sentiment)</b><br/>"
                                 f"   - Analyze {len(negative_reviews):,} negative reviews to identify recurring problems<br/>"
                                 f"   - Implement fixes in next 2-3 release cycles<br/>"
                                 f"   - Set up monitoring for sentiment trend changes")
    
    if pos_pct >= 50:
        recommendations.append(f"<b>2. Amplify Positive Experiences ({pos_pct:.1f}% positive sentiment)</b><br/>"
                             f"   - Feature positive reviews in marketing materials<br/>"
                             f"   - Encourage satisfied users to share reviews<br/>"
                             f"   - Document and replicate what's working well")
    
    if neu_pct >= 25:
        recommendations.append(f"<b>3. Convert Neutral Users to Advocates ({neu_pct:.1f}% neutral sentiment)</b><br/>"
                             f"   - Identify missing features from neutral reviews<br/>"
                             f"   - Improve onboarding and user guidance<br/>"
                             f"   - Implement loyalty programs or engagement features")
    
    recommendations.append("<b>4. Continuous Monitoring</b><br/>"
                         "   - Track sentiment trends weekly<br/>"
                         "   - Monitor rating changes per app version<br/>"
                         "   - Set up alerts for sudden negative sentiment spikes")
    
    for rec in recommendations:
        story.append(Paragraph(rec, body_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer