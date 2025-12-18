# Upload Model to Hugging Face Hub

## Cara Upload Model RoBERTa ke Hugging Face (Agar Bisa Dipakai di Streamlit Cloud)

### Step 1: Install Hugging Face CLI
```bash
pip install huggingface_hub
```

### Step 2: Login ke Hugging Face
```bash
huggingface-cli login
```
Masukkan token dari https://huggingface.co/settings/tokens

### Step 3: Upload Model
```python
from huggingface_hub import HfApi

api = HfApi()

# Upload semua file di saved_model/ ke repo HF
api.upload_folder(
    folder_path="saved_model",
    repo_id="rkzzone/roberta-sentiment-playstore",  # Ganti dengan username HF kamu
    repo_type="model",
    commit_message="Upload fine-tuned RoBERTa for Play Store sentiment analysis"
)
```

### Step 4: Verifikasi
- Buka https://huggingface.co/rkzzone/roberta-sentiment-playstore
- Pastikan semua file terupload (config.json, model.safetensors, tokenizer files)

### Step 5: Update utils.py
Ganti `model_name` di utils.py dengan repo HF kamu:
```python
model_name = "USERNAME/roberta-sentiment-playstore"  # Ganti USERNAME
```

---

## Alternative: Pakai Base Model Saja (Quick Fix)

Jika tidak mau upload ke HF, kode sekarang akan otomatis fallback ke `roberta-base`.
⚠️ **WARNING**: Model base TIDAK fine-tuned, hasil sentiment analysis akan kurang akurat!

Untuk production, **sangat disarankan** upload model yang sudah di-train ke Hugging Face Hub.
