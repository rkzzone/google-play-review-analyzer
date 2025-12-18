"""
Quick script to upload Indonesian sentiment model to HuggingFace Hub
Run this in Jupyter/Colab or any Python environment with internet access
"""

from huggingface_hub import HfApi, login
import os

# Configuration
USERNAME = "rkkzone"  # Change this to your HuggingFace username
REPO_NAME = "roberta-sentiment-indonesian-playstore"
MODEL_FOLDER = "saved_model_id"
TOKEN = ""  # Paste your token here or will prompt

# Step 1: Login
print("=" * 70)
print("🚀 UPLOADING INDONESIAN SENTIMENT MODEL TO HUGGINGFACE HUB")
print("=" * 70)

if not TOKEN:
    print("\n📝 Get your token from: https://huggingface.co/settings/tokens")
    TOKEN = input("Paste your HuggingFace token: ").strip()

try:
    login(token=TOKEN)
    print("✅ Login successful!")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

# Step 2: Upload model
api = HfApi()
repo_id = f"{USERNAME}/{REPO_NAME}"

print(f"\n📦 Repository: {repo_id}")
print(f"📁 Model folder: {MODEL_FOLDER} (~475MB)")
print("\nFiles to upload:")
for file in os.listdir(MODEL_FOLDER):
    size = os.path.getsize(os.path.join(MODEL_FOLDER, file)) / (1024**2)
    print(f"  - {file} ({size:.2f} MB)")

confirm = input("\n⚠️  Proceed with upload? (y/n): ").strip().lower()
if confirm != 'y':
    print("❌ Upload cancelled")
    exit(0)

try:
    # Create repo
    print(f"\n📦 Creating repository: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    
    # Upload folder
    print(f"⬆️  Uploading files from {MODEL_FOLDER}/...")
    print("⏳ This may take 5-10 minutes for ~475MB...")
    
    api.upload_folder(
        folder_path=MODEL_FOLDER,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Indonesian sentiment model trained on SMSA dataset"
    )
    
    print(f"\n✅ Upload successful!")
    print(f"🔗 Model available at: https://huggingface.co/{repo_id}")
    
    # Create README
    print("\n📝 Creating README.md...")
    readme_content = f"""---
language: id
license: mit
tags:
- sentiment-analysis
- indobert
- indonesian
- google-play-reviews
- text-classification
datasets:
- smsa
metrics:
- accuracy
- f1
---

# {REPO_NAME}

Fine-tuned RoBERTa model for **Indonesian** sentiment analysis on Google Play Store reviews.

## Model Description

This model performs 3-class sentiment classification:
- **Positive** (label: 2) 😊
- **Neutral** (label: 1) 😐
- **Negative** (label: 0) 😞

## Training Data

- **Dataset**: SMSA (Sentiment Analysis on Indonesian Movie Reviews)
- **Language**: Indonesian (Bahasa Indonesia)
- **Size**: 11,000 reviews
  - Positive: 6,416 reviews
  - Negative: 3,436 reviews
  - Neutral: 1,148 reviews
- **Domain**: App reviews (Google Play Store)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example Indonesian review
text = "Aplikasi bagus sekali! Sangat direkomendasikan."

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(predictions, dim=-1).item()

sentiment_labels = {{0: "Negative", 1: "Neutral", 2: "Positive"}}
print(f"Sentiment: {{sentiment_labels[sentiment]}}")
print(f"Confidence: {{predictions[0][sentiment].item():.4f}}")
```

## Model Performance

Trained with:
- **Base Model**: IndoBERT / RoBERTa
- **Training Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens

Evaluated on test split with weighted F1 score and accuracy metrics.

## Intended Use

This model is optimized for:
- ✅ Indonesian language app reviews
- ✅ Google Play Store sentiment analysis
- ✅ Customer feedback classification
- ✅ Review monitoring and analytics

## Limitations

- Optimized for app review domain
- May not generalize well to formal Indonesian text
- Best performance on colloquial Indonesian reviews
- Sensitive to slang and informal language variations

## Applications

- **App Analytics**: Analyze user sentiment in Indonesian app reviews
- **Customer Insights**: Monitor feedback trends and satisfaction
- **Review Filtering**: Identify negative reviews for priority response
- **Market Research**: Understand Indonesian user preferences

## Citation

```bibtex
@misc{{roberta_sentiment_indonesian_playstore,
  author = {{{USERNAME}}},
  title = {{Indonesian Sentiment Analysis for Google Play Reviews}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## License

MIT License

## Related Models

- English version: [rkkzone/roberta-sentiment-playstore](https://huggingface.co/rkkzone/roberta-sentiment-playstore)

---

**Built with ❤️ for Indonesian app developers**
"""
    
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add comprehensive README"
    )
    
    print("✅ README created successfully!")
    print(f"\n🎉 All done! Visit your model at:")
    print(f"   https://huggingface.co/{repo_id}")
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Verify your token has write permissions")
    print("3. Ensure the model folder exists and contains all files")
    exit(1)
