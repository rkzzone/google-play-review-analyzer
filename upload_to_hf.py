from huggingface_hub import HfApi, login
import os

# Step 1: Login ke Hugging Face
print("=" * 60)
print("UPLOAD MODEL TO HUGGING FACE HUB")
print("=" * 60)
print("\n1. Buat akun di https://huggingface.co jika belum punya")
print("2. Buat token di https://huggingface.co/settings/tokens")
print("3. Paste token di bawah ini:\n")

token = input("Paste your Hugging Face token: ").strip()

if not token:
    print("❌ Token tidak boleh kosong!")
    exit(1)

try:
    login(token=token)
    print("✅ Login berhasil!")
except Exception as e:
    print(f"❌ Login gagal: {e}")
    exit(1)

# Step 2: Tentukan repo name
print("\n" + "=" * 60)
username = input("Masukkan username Hugging Face kamu: ").strip()

# Pilih model yang akan di-upload
print("\nPilih model yang akan di-upload:")
print("1. English Model (saved_model/)")
print("2. Indonesian Model (saved_model_id/)")
choice = input("Pilihan (1/2): ").strip()

if choice == "1":
    model_folder = "saved_model"
    default_repo_name = "roberta-sentiment-playstore"
    description = "Fine-tuned RoBERTa for English sentiment analysis on Google Play Store reviews"
elif choice == "2":
    model_folder = "saved_model_id"
    default_repo_name = "indobert-sentiment-playstore"
    description = "Fine-tuned IndoBERT for Indonesian sentiment analysis on Google Play Store reviews"
else:
    print("❌ Pilihan tidak valid!")
    exit(1)

repo_name = input(f"Nama repository [{default_repo_name}]: ").strip() or default_repo_name
repo_id = f"{username}/{repo_name}"

print(f"\nRepo akan dibuat di: https://huggingface.co/{repo_id}")
confirm = input("Lanjutkan? (y/n): ").strip().lower()

if confirm != 'y':
    print("❌ Upload dibatalkan")
    exit(0)

# Step 3: Upload model
api = HfApi()

print("\n" + "=" * 60)
print("UPLOADING MODEL...")
print("=" * 60)

try:
    # Create repo if not exists
    print(f"📦 Membuat repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    
    # Upload folder
    print(f"⬆️  Uploading files dari {model_folder}/...")
    api.upload_folder(
        folder_path=model_folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {description}"
    )
    
    print(f"\n✅ Upload berhasil!")
    print(f"🔗 Model tersedia di: https://huggingface.co/{repo_id}")
    
    # Update README
    print("\n📝 Updating README...")
    readme_content = f"""---
language: {'en' if choice == '1' else 'id'}
license: mit
tags:
- sentiment-analysis
- {'roberta' if choice == '1' else 'indobert'}
- google-play-reviews
- text-classification
datasets:
- google-play-store-reviews
metrics:
- accuracy
- f1
---

# {repo_name}

{description}

## Model Description

This model is fine-tuned for 3-class sentiment classification:
- **Positive** (label: 2)
- **Neutral** (label: 1)  
- **Negative** (label: 0)

## Training Data

Trained on Google Play Store reviews in {'English' if choice == '1' else 'Indonesian'} language.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "{'Great app! Highly recommended.' if choice == '1' else 'Aplikasi bagus sekali! Sangat direkomendasikan.'}"
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

Evaluated on test set with metrics:
- Accuracy
- F1 Score (weighted)
- Precision & Recall per class

## Intended Use

This model is designed for sentiment analysis of {'English' if choice == '1' else 'Indonesian'} language app reviews from Google Play Store.

## Limitations

- Optimized for app review domain
- May not generalize well to other text types
- Best performance on reviews similar to training data

## Citation

```
@misc{{{repo_name.replace('-', '_')},
  author = {{{username}}},
  title = {{{repo_name}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""
    
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
        commit_message="Upload fine-tuned RoBERTa for Google Play Store sentiment analysis"
    )
    
    print("\n" + "=" * 60)
    print("✅ UPLOAD BERHASIL!")
    print("=" * 60)
    print(f"\n🔗 Model URL: https://huggingface.co/{repo_id}")
    print(f"\n📝 NEXT STEP:")
    print(f"   Update utils.py line ~350:")
    print(f'   model_name = "{repo_id}"')
    print(f"\n   Lalu commit & push ke GitHub")
    
except Exception as e:
    print(f"\n❌ Upload gagal: {e}")
    print("\nTroubleshooting:")
    print("1. Pastikan folder saved_model/ ada dan berisi file model")
    print("2. Pastikan token HF kamu punya write permission")
    print("3. Cek koneksi internet")
