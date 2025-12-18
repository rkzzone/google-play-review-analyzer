# 🧪 Testing Guide - Indonesian Apps

Panduan testing untuk verifikasi Indonesian sentiment model di production.

## 📱 Recommended Indonesian Apps for Testing

### E-Commerce Apps
1. **Tokopedia** (`com.tokopedia.tkpd`)
   - Expected: Mixed sentiments (positive features, some complaint about bugs)
   - High review count (~10M+ reviews)
   
2. **Shopee** (`com.shopee.id`)
   - Expected: Generally positive with delivery complaints
   - Very high engagement

3. **Bukalapak** (`com.bukalapak.android`)
   - Expected: Balanced sentiments
   - Medium review count

### Transportation/Delivery
4. **Gojek** (`com.gojek.app`)
   - Expected: Mixed (driver issues, app performance)
   - Huge review base

5. **Grab** (`com.grabtaxi.passenger`)
   - Expected: Similar to Gojek
   - Indonesian-specific reviews

### Banking/Finance
6. **BCA mobile** (`com.bca`)
   - Expected: Mostly positive, some technical issues
   - Indonesian banking app

7. **GoPay** (`com.gopay`)
   - Expected: Payment issues, convenience feedback

### Social Media
8. **TikTok** 
   - Expected: Very mixed sentiments
   - Bahasa Indonesia slang heavy

## ✅ Testing Checklist

### 1. Search Functionality
- [ ] Search "Tokopedia" → Returns correct app
- [ ] Search "Gojek" → Shows proper app card
- [ ] App details displayed correctly (rating, developer, installs)

### 2. Review Scraping
- [ ] Scrape 100 reviews from Tokopedia
- [ ] Reviews are in Indonesian language
- [ ] Date range filter works properly
- [ ] No English reviews mixed in

### 3. Sentiment Analysis
Test these Indonesian phrases manually or check in scraped data:

**Positive Examples:**
- "Aplikasi bagus banget! Mudah digunakan" → Should be **Positive**
- "Mantap jiwa, recommended deh!" → Should be **Positive**
- "Fitur lengkap, pengiriman cepat" → Should be **Positive**

**Negative Examples:**
- "Aplikasi sering crash, mohon diperbaiki" → Should be **Negative**
- "Kecewa, uang hilang, customer service lambat" → Should be **Negative**
- "Lemot banget, gak bisa dibuka" → Should be **Negative**

**Neutral Examples:**
- "Biasa aja sih, standar" → Should be **Neutral**
- "Lumayan lah untuk aplikasi gratis" → Should be **Neutral**

### 4. Topic Modeling
Check if topics show **Indonesian keywords**, not English:

**Expected Indonesian Topics:**
- "pengiriman, cepat, barang" (delivery topic)
- "harga, murah, promo" (price topic)
- "aplikasi, error, crash" (technical topic)
- "customer, service, respon" (support topic)

**Bad (shouldn't happen):**
- English words like "delivery", "price", "error" dominating topics

### 5. Visualizations
- [ ] Sentiment pie chart shows correct distribution
- [ ] Timeline shows Indonesian date format (if applicable)
- [ ] Topic bar charts readable
- [ ] Negative keywords extraction working

### 6. PDF Report
- [ ] PDF generates successfully
- [ ] Indonesian text renders properly (no encoding issues)
- [ ] Charts embedded correctly
- [ ] Insights/recommendations relevant

## 🐛 Common Issues to Watch For

### Model Loading
**Issue**: Model fails to download from HuggingFace
**Fix**: Check internet, HF model is public, repo name correct

### Indonesian Text Rendering
**Issue**: Indonesian characters broken (è, ô, etc.)
**Fix**: UTF-8 encoding in PDF generation

### Slang Not Recognized
**Issue**: "gak", "banget", "mantap" not normalized
**Fix**: Check slang_dict applied in preprocessing

### English Reviews Mixed In
**Issue**: Some English reviews appear
**Fix**: Verify lang='id', country='id' in scraping

### Topic Modeling in English
**Issue**: Topics show English words
**Fix**: Confirm Indonesian stopwords loaded, multilingual embeddings used

## 📊 Expected Metrics

### Tokopedia Test (100 reviews)
- **Sentiment Distribution**: ~60% Positive, ~25% Negative, ~15% Neutral
- **Avg Rating**: ~4.2-4.5
- **Common Topics**: "pengiriman", "harga", "barang", "seller"
- **Negative Keywords**: "lambat", "rusak", "kecewa", "bohong"

### Gojek Test (100 reviews)
- **Sentiment Distribution**: ~50% Positive, ~35% Negative, ~15% Neutral
- **Avg Rating**: ~4.0-4.3
- **Common Topics**: "driver", "orderan", "aplikasi", "promo"
- **Negative Keywords**: "cancel", "lama", "error", "suspend"

## 🎯 Success Criteria

✅ **Search**: Indonesian apps found correctly (90%+ match rate)
✅ **Scraping**: 100% Indonesian reviews (no English mixed)
✅ **Sentiment**: 80%+ accuracy on manual spot-check (20 random reviews)
✅ **Topics**: All Indonesian keywords (no English domination)
✅ **Performance**: Model loads in <30 seconds, analysis <2 minutes
✅ **PDF**: Generates successfully with proper Indonesian text

## 🚀 Quick Test Script

```python
# Test Indonesian sentiment directly in Python console
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "rkkzone/roberta-sentiment-indonesian-playstore"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

test_reviews = [
    "Aplikasi bagus banget, sangat membantu!",
    "Kecewa, aplikasi sering error dan lambat",
    "Biasa aja sih, gak ada yang spesial"
]

for review in test_reviews:
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"{review[:50]}... → {labels[pred]}")
```

## 📝 Test Report Template

After testing, document results:

```markdown
# Indonesian Model Test Report

**Date**: [Date]
**Tester**: [Your Name]
**Dashboard URL**: https://google-play-review-analyzer.streamlit.app

## Test Results

### App: Tokopedia
- Reviews Scraped: ✅ 100 Indonesian reviews
- Sentiment Accuracy: ✅ 85% (17/20 spot-check correct)
- Topics Quality: ✅ All Indonesian keywords
- Performance: ✅ Analysis completed in 1m 30s
- PDF Generated: ✅ No encoding issues

### Issues Found
1. [List any issues]
2. [Solutions applied]

### Overall Assessment
✅ PASS / ❌ FAIL

**Notes**: [Additional observations]
```

---

**Ready to deploy!** 🚀

Visit your Streamlit dashboard and monitor the rebuild process.
