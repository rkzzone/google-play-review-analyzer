# PDF Report Templates

## Cara Menggunakan Template

### Format Template
- **Format**: PDF 16:9 landscape (11 x 6.1875 inch)
- **File**: `report_template.pdf` (5 halaman)

### Struktur Halaman

#### Page 1: Cover
- Area kosong untuk title app (center, top 40%)
- Area untuk metadata (center, bottom)

#### Page 2: Executive Summary
**Placeholder positions:**
- KPI Cards: Top area (3 boxes horizontal)
- Summary Chart: Center area (pie/donut chart)
- Insights: Bottom area (bullet points)

#### Page 3: Sentiment Analysis
**Placeholder positions:**
- Sentiment Breakdown Table: Left 40%
- Timeline Chart: Right 60%

#### Page 4: Top Topics
**Placeholder positions:**
- Topic Bar Chart: Left 60%
- Top Reviews: Right 40%

#### Page 5: Recommendations
**Text areas untuk insights**

---

## Cara Membuat Template

### Opsi A: PowerPoint/Keynote
1. Buat presentasi 16:9 dengan desain profesional
2. Buat 5 slide sesuai struktur di atas
3. Tambahkan placeholder text atau shape untuk marking area
4. Export as PDF → Save sebagai `report_template.pdf`

### Opsi B: Canva/Figma
1. Canvas size: 1920 x 1080 px (16:9)
2. Design 5 frames dengan branding Anda
3. Export each frame as PDF page
4. Merge jadi satu PDF

### Opsi C: Adobe Illustrator
1. Artboard: 11 x 6.1875 inch landscape
2. Design dengan layer terpisah
3. Export as multi-page PDF

---

## Coordinate Mapping (untuk customization)

Edit di `utils.py` bagian `TEMPLATE_CONFIG`:

```python
TEMPLATE_CONFIG = {
    "page_2": {
        "kpi_area": (50, 400, 750, 500),  # (x, y, width, height) in points
        "chart_area": (50, 150, 400, 350),
        "insights_area": (500, 150, 250, 350)
    },
    # ... dst
}
```

---

## Default Template
Jika tidak ada `report_template.pdf`, sistem akan generate layout default dengan:
- Background putih
- Header biru (#1f77b4)
- Professional typography
