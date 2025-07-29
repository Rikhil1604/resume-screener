from xhtml2pdf import pisa
from io import BytesIO

def convert_html_to_pdf(feedback_text, job_title, category, ats_score, freshness, jd_keywords=None):
    """Generate a styled PDF with structured AI feedback and metadata."""

    jd_keywords_str = ", ".join(jd_keywords) if jd_keywords else "N/A"

    html_template = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                padding: 30px;
                color: #333;
            }}
            h1 {{
                color: #003366;
                font-size: 22px;
                border-bottom: 2px solid #003366;
                padding-bottom: 6px;
            }}
            h2 {{
                color: #004488;
                font-size: 18px;
                margin-top: 1em;
            }}
            .section {{
                margin-top: 20px;
            }}
            .score {{
                font-weight: bold;
                font-size: 18px;
                color: #0066cc;
                margin-top: 1em;
            }}
            .metadata {{
                background-color: #f1f5f9;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .metadata p {{
                margin: 4px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Resume Evaluation Report</h1>

        <div class="metadata">
            <p><strong>🎯 Target Role:</strong> {job_title}</p>
            <p><strong>🧠 Predicted Category:</strong> {category}</p>
            <p><strong>📊 ATS Match Score:</strong> {ats_score}/100</p>
            <p><strong>📅 Resume Freshness:</strong> {freshness}</p>
            <p><strong>📌 Extracted JD Keywords:</strong> {jd_keywords_str}</p>
        </div>

        <div class="section">
            {format_feedback_as_html(feedback_text)}
        </div>
    </body>
    </html>
    """

    pdf_output = BytesIO()
    pisa.CreatePDF(src=html_template, dest=pdf_output)
    return pdf_output.getvalue()


def format_feedback_as_html(text):
    """Convert feedback string to structured HTML with headings."""
    lines = text.strip().split("\n")
    html_parts = []

    for line in lines:
        line = line.strip()
        if line.lower().startswith("score:"):
            html_parts.append(f'<p class="score">{line}</p>')
        elif any(kw in line.lower() for kw in ["strength", "area", "improve"]):
            html_parts.append(f"<h2>{line}</h2>")
        else:
            html_parts.append(f"<p>{line}</p>")

    return "\n".join(html_parts)
