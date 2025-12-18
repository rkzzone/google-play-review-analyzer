# 📱 App Review Sentiment & Topic Intelligence Dashboard

Dashboard interaktif untuk menganalisis sentimen dan topik dari review aplikasi Google Play Store Indonesia menggunakan **IndoBERT** dan **BERTopic**.

## 🌟 Fitur

- **Real-time Scraping**: Ambil review bahasa Indonesia langsung dari Google Play Store
- **Sentiment Analysis**: Klasifikasi sentimen menggunakan model IndoBERT fine-tuned untuk bahasa Indonesia
- **Topic Modeling**: Ekstraksi topik otomatis dengan BERTopic (multilingual support)
- **Interactive Dashboard**: Visualisasi data yang komprehensif dan adaptif
- **Professional PDF Reports**: Export hasil analisis dalam format PDF konsultan 16:9

## 🛠️ Teknologi

- **Frontend**: Streamlit
- **ML Models**: 
  - **IndoBERT** (Sentiment Analysis - Indonesian)
  - **BERTopic** with Multilingual Embeddings (Topic Modeling)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Kaleido
- **PDF Generation**: ReportLab, PyPDF2
- **Scraping**: google-play-scraper

## 📊 Model Information

- **Model**: [rkkzone/indobert-sentiment-indonesian-playstore](https://huggingface.co/rkkzone/indobert-sentiment-indonesian-playstore)
- **Language**: Indonesian (Bahasa Indonesia)
- **Training Dataset**: SMSA (11,000 Indonesian reviews)
- **Classes**: Positive, Neutral, Negative
- **Base Model**: IndoBERT (`indobenchmark/indobert-base-p1`)

## 📋 Prerequisites

- Python 3.8+
- pip

## 🚀 Installation

1. Clone repository ini:
```bash
git clone https://github.com/YOUR_USERNAME/nlp-ki.git
cd nlp-ki
```

2. Buat virtual environment (opsional tapi direkomendasikan):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

Dataset yang digunakan untuk training: `googleplaystore_user_reviews.csv`

Letakkan dataset di folder `dataset/`:
```
nlp-ki/
├── dataset/
│   └── googleplaystore_user_reviews.csv
```

## 🎯 Training Model

Untuk melatih model sentiment analysis sendiri, jalankan notebook:

```bash
jupyter notebook training_sentiment.ipynb
```

Atau jalankan semua sel secara berurutan. Model akan tersimpan di folder `saved_model/`.

## 🎨 Menjalankan Dashboard

```bash
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

## 📖 Cara Penggunaan

1. **Pilih Aplikasi**:
   - Cari aplikasi by nama, atau
   - Masukkan App ID langsung (contoh: `com.whatsapp`)

2. **Konfigurasi Scraping**:
   - Pilih jumlah review atau rentang tanggal
   - Klik "Fetch Reviews"

3. **Analisis**:
   - Lihat distribusi sentimen
   - Eksplorasi topik yang diekstrak
   - Analisis tren temporal
   - Review kata-kata yang sering muncul

4. **Export**:
   - Download hasil analisis dalam format CSV

## 📁 Struktur Project

```
nlp-ki/
├── app.py                          # Streamlit dashboard
├── utils.py                        # Utility functions
├── training_sentiment.ipynb        # Notebook untuk training model
├── requirements.txt                # Python dependencies
├── dataset/
│   └── googleplaystore_user_reviews.csv
├── saved_model/                    # Trained model files (setelah training)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ...
└── README.md
```

## 🔑 Konfigurasi Model

Model sentiment analysis menggunakan:
- **Base Model**: `roberta-base`
- **Labels**: 3 kelas (Negative, Neutral, Positive)
- **Max Length**: 128 tokens
- **Training**: 3 epochs dengan balanced dataset

## ⚠️ Catatan Penting

1. **Rate Limiting**: Google Play Scraper memiliki rate limiting. Jika scraping gagal, tunggu beberapa saat sebelum mencoba lagi.

2. **App ID**: Beberapa aplikasi mungkin tidak memiliki App ID yang valid dari hasil search. Gunakan App ID langsung untuk hasil terbaik.

3. **Model Files**: Folder `saved_model/` tidak di-commit ke Git karena ukurannya besar. Anda perlu train model sendiri atau download dari release.

## 🐛 Troubleshooting

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: Model not found
Pastikan sudah menjalankan `training_sentiment.ipynb` atau download model dari release.

### Error: No reviews found
- Periksa koneksi internet
- Gunakan App ID yang valid
- Coba aplikasi lain yang populer

## 📝 License

MIT License

## 👤 Author

Muhammad - NLP & AI Project

## 🙏 Acknowledgments

- Hugging Face Transformers
- Google Play Scraper
- Streamlit Community
- BERTopic
