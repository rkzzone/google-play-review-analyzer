# Template Configuration Example
# Copy this to utils.py and customize coordinates based on your template

TEMPLATE_CONFIG = {
    # Page 2: Executive Summary
    "page_2_executive": {
        # KPI Boxes positions (x, y, width, height) in points
        "kpi_total_reviews": (50, 420, 230, 60),
        "kpi_avg_rating": (300, 420, 230, 60),
        "kpi_sentiment": (550, 420, 230, 60),
        
        # Chart positions
        "chart_donut": {
            "position": (50, 150),   # Bottom-left (x, y)
            "size": (280, 250)       # (width, height) in points
        },
        "chart_timeline": {
            "position": (380, 150),
            "size": (400, 250)
        }
    },
    
    # Page 3: Sentiment Deep Dive
    "page_3_sentiment": {
        "sentiment_table": (50, 320, 350, 180),
        "chart_breakdown": {
            "position": (450, 150),
            "size": (350, 300)
        }
    },
    
    # Page 4: Topics
    "page_4_topics": {
        "chart_topics_bar": {
            "position": (50, 120),
            "size": (500, 350)
        },
        "top_reviews_area": (600, 120, 180, 350)
    },
    
    # Page 5: Recommendations
    "page_5_recommendations": {
        "text_area_1": (50, 350, 700, 80),
        "text_area_2": (50, 250, 700, 80),
        "text_area_3": (50, 150, 700, 80)
    }
}

# Coordinate mapping guide:
# 
# Points to Inches: points / 72 = inches
# Example: 792 points = 11 inches (width of 16:9 landscape)
# 
# Origin: Bottom-left corner of page
# X-axis: Left to right (0 to 792 points for 11")
# Y-axis: Bottom to top (0 to 446 points for 6.1875")
# 
# Finding coordinates:
# 1. Open template PDF in PDF editor (Adobe Acrobat, PDFelement)
# 2. Use ruler tool to measure position
# 3. Or use selection tool to see coordinates
# 4. Convert to points (if in inches: inches * 72)
