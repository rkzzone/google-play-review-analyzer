# Quick Start: PDF Template

## 🚀 Cara Tercepat (5 Menit)

### **Opsi A: Pakai Template Default (No Setup)**
Tidak perlu apa-apa! Sistem sudah include visualisasi chart.
- Langsung klik "Download Professional Report (PDF)"
- PDF akan generate otomatis dengan chart embedded

---

### **Opsi B: Gunakan Template Custom PowerPoint**

**Step 1:** Buat PowerPoint 16:9
- Slide 1: Cover (title placeholder)
- Slide 2-5: Content dengan placeholder `[CHART_XXX]`

**Step 2:** Export as PDF
```
File → Export → Create PDF → Save as "report_template.pdf"
```

**Step 3:** Copy ke folder
```
nlp-ki/templates/report_template.pdf
```

**Step 4:** Done! 
Sistem otomatis detect dan pakai template Anda.

---

## 📝 Placeholder Tags yang Didukung

Tulis text ini di template PowerPoint Anda, akan auto-replace:

### **Data Placeholders:**
- `[APP_NAME]` → Nama aplikasi
- `[DEVELOPER]` → Developer name
- `[DATE]` → Tanggal generate
- `[TOTAL_REVIEWS]` → Jumlah reviews
- `[AVG_RATING]` → Rating rata-rata
- `[SENTIMENT]` → Sentiment dominant

### **Chart Placeholders:**
- `[CHART_DONUT]` → Pie chart sentiment distribution
- `[CHART_TIMELINE]` → Line chart trend over time
- `[CHART_TOPICS]` → Bar chart top topics
- `[CHART_VERSION]` → Stacked bar version analysis

### **Insight Placeholders:**
- `[INSIGHT_1]` → Key insight pertama
- `[INSIGHT_2]` → Key insight kedua
- `[RECOMMENDATION_1]` → Rekomendasi strategis

---

## ✅ Contoh Slide PowerPoint

### **Slide 2 Example:**

```
┌─────────────────────────────────────────────┐
│   Executive Summary                          │
├─────────────────────────────────────────────┤
│                                             │
│  📊 Total Reviews    ⭐ Avg Rating    💭 Sentiment │
│  [TOTAL_REVIEWS]    [AVG_RATING]    [SENTIMENT]  │
│                                             │
│  ┌─────────────┐    ┌──────────────────┐  │
│  │             │    │                  │  │
│  │[CHART_DONUT]│    │  [CHART_TIMELINE]│  │
│  │             │    │                  │  │
│  └─────────────┘    └──────────────────┘  │
│                                             │
│  Key Insight: [INSIGHT_1]                  │
└─────────────────────────────────────────────┘
```

**Catatan:** Shapes/text boxes dengan tag placeholder akan di-replace dengan chart PNG.

---

## 🎨 Design Tips

1. **Background:** Pakai warna brand Anda
2. **Font:** Helvetica/Arial untuk readability
3. **Colors:** 
   - Positive: Green (#2ecc71)
   - Neutral: Gray (#95a5a6)
   - Negative: Red (#e74c3c)
4. **Logo:** Taruh di top-left/top-right setiap slide
5. **Footer:** Company name + page number

---

## 🔧 Advanced: Coordinate-Based

Jika mau kontrol penuh tanpa placeholder text:

1. Buat template **blank** (hanya background design)
2. Edit `TEMPLATE_CONFIG` di `utils.py`
3. Specify exact coordinates untuk setiap element

**Contoh:**
```python
"page_2_executive": {
    "chart_donut": {
        "position": (50, 200),  # 50pt from left, 200pt from bottom
        "size": (300, 250)       # 300pt wide, 250pt tall
    }
}
```

4. Chart akan overlay di koordinat exact tersebut

---

## ❓ Troubleshooting

**Q: Template tidak terdeteksi?**
- Cek nama file: harus `report_template.pdf` (lowercase)
- Cek folder: `nlp-ki/templates/`
- Cek page count: minimal 5 pages

**Q: Chart tidak muncul di template?**
- Pastikan install kaleido: `pip install kaleido`
- Check placeholder tag exact match (case-sensitive)

**Q: Coordinate salah?**
- Pakai PDF editor untuk measure position
- Origin: bottom-left (0,0)
- Unit: points (72 points = 1 inch)

---

## 📦 Files yang Dibutuhkan

```
nlp-ki/
├── templates/
│   ├── report_template.pdf     ← Your PowerPoint export (OPTIONAL)
│   ├── README.md               ← Documentation
│   └── TUTORIAL.md             ← This file
├── app.py
├── utils.py                    ← Contains TEMPLATE_CONFIG
└── requirements.txt            ← PyPDF2, kaleido installed
```

---

## 🎯 Next Action

**Untuk coba sekarang:**

1. Tidak perlu template → Sistem pakai default (sudah ada charts!)
2. Download report dari app → Lihat hasil
3. Kalau mau custom → Buat PowerPoint → Export PDF → Upload

**Atau kalau mau langsung test template system, saya bisa buatkan contoh template sederhana?**
