# Tutorial: Menggunakan Template PDF untuk Report

## **Workflow Sederhana**

### **1. Buat Template di PowerPoint**

**Langkah-langkah:**

1. **Buka PowerPoint** → New Presentation → Slide Size: **16:9 (Widescreen)**

2. **Slide 1 - Cover Page**
   ```
   Layout: Title Slide
   - Add company logo (top left)
   - Title placeholder: "App Review Analysis Report"
   - Subtitle: "[APP_NAME]"  ← ini akan di-replace
   - Footer: "Generated on [DATE]" ← ini akan di-replace
   ```

3. **Slide 2 - Executive Summary**
   ```
   Layout: Title and Content
   - Title: "Executive Summary"
   - Insert 3 text boxes untuk KPI cards:
     * Box 1 (left): "Total Reviews: [VALUE]"
     * Box 2 (center): "Avg Rating: [VALUE]"
     * Box 3 (right): "Sentiment: [VALUE]"
   - Insert placeholder shape (rectangle) untuk chart
     * Position: Center-left
     * Size: ~400x300 px
     * Add text inside: "[CHART_SENTIMENT]"
   ```

4. **Slide 3 - Sentiment Trends**
   ```
   - Title: "Sentiment Analysis Over Time"
   - Placeholder untuk timeline chart
     * Add shape dengan text: "[CHART_TIMELINE]"
     * Size: ~700x350 px
   ```

5. **Slide 4 - Top Topics**
   ```
   - Title: "Key Discussion Topics"
   - Placeholder: "[CHART_TOPICS]"
   - Text area untuk insights
   ```

6. **Slide 5 - Recommendations**
   ```
   - Title: "Strategic Recommendations"
   - Bullet points dengan placeholder:
     * "[RECOMMENDATION_1]"
     * "[RECOMMENDATION_2]"
     * "[RECOMMENDATION_3]"
   ```

7. **Export sebagai PDF:**
   - File → Export → PDF
   - Save as: `report_template.pdf`
   - Copy ke folder: `nlp-ki/templates/`

---

### **2. Sistem Akan Otomatis Replace**

Ketika user klik "Download PDF Report", sistem akan:

1. **Load template** dari `templates/report_template.pdf`
2. **Generate charts** dari data (Plotly → PNG)
3. **Replace placeholders:**
   - `[APP_NAME]` → `"Instagram"`
   - `[DATE]` → `"December 18, 2025"`
   - `[VALUE]` → Data aktual (500, 4.02, etc.)
   - `[CHART_SENTIMENT]` → Image chart donut
   - `[CHART_TIMELINE]` → Image line chart
   - `[CHART_TOPICS]` → Image bar chart
   - `[RECOMMENDATION_X]` → Insight text

4. **Merge** semua jadi satu PDF final

---

### **3. Coordinate-Based Approach (Advanced)**

Jika tidak ingin pakai placeholder text, bisa gunakan koordinat:

**Cara:**
1. Template PDF Anda **kosong/polos** (hanya design background & header)
2. Edit `TEMPLATE_CONFIG` di `utils.py`:

```python
TEMPLATE_CONFIG = {
    "page_2_executive": {
        # Koordinat dalam points (72 points = 1 inch)
        # Origin: bottom-left corner
        "kpi_boxes": [
            (50, 400, 200, 80),   # (x, y, width, height) - Box 1
            (280, 400, 200, 80),  # Box 2
            (510, 400, 200, 80)   # Box 3
        ],
        "chart_sentiment": {
            "position": (50, 100),  # (x, y) dari bottom-left
            "size": (300, 250)      # (width, height)
        }
    }
}
```

3. Script akan **overlay** chart dan text di koordinat tersebut

---

## **Testing Template**

### **Cek Template Valid:**

Jalankan test script:

```python
import os
from PyPDF2 import PdfReader

template_path = "templates/report_template.pdf"

if os.path.exists(template_path):
    reader = PdfReader(template_path)
    print(f"✅ Template found: {len(reader.pages)} pages")
    
    # Check page size (harus 16:9)
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    ratio = width / height
    
    print(f"Page size: {width/72:.1f}\" x {height/72:.1f}\"")
    print(f"Aspect ratio: {ratio:.2f} (should be ~1.78 for 16:9)")
else:
    print("❌ Template not found")
```

---

## **Contoh Template Sederhana**

Jika mau coba cepat tanpa PowerPoint:

1. Download template gratis dari Canva:
   - https://www.canva.com/templates/?query=presentation+16:9
   - Filter: Presentation, 16:9
   - Pilih yang minimalis/professional

2. Edit sesuai kebutuhan
3. Download as PDF
4. Upload ke `templates/`

---

## **Fallback ke Default**

Jika `report_template.pdf` tidak ada, sistem akan:
- Generate layout default dengan reportlab
- Include semua visualisasi
- Professional styling (biru/putih)

**Keuntungan:** Tetap bisa generate report tanpa template custom!

---

## **Next Steps**

Setelah template siap, test dengan:
1. Analisis app (misalnya Instagram)
2. Klik "Download Professional Report (PDF)"
3. Cek hasil PDF - apakah chart overlay benar
4. Adjust koordinat di `TEMPLATE_CONFIG` jika perlu
