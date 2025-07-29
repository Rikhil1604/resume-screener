# resume_parser.py

import fitz  # PyMuPDF
import re
from datetime import datetime

def extract_text_from_pdf(file):
    """Extract raw text from a PDF file using PyMuPDF (fitz)."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def estimate_resume_freshness(resume_text):
    """Rudimentary logic to guess how recently the resume was updated."""
    # Extract years or years with months
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', resume_text)
    current_year = datetime.now().year

    if years:
        latest_year = max(map(int, years))
        diff = current_year - latest_year
        if diff <= 1:
            return "Updated Recently"
        elif diff <= 3:
            return "Updated within last 2-3 years"
        else:
            return "Outdated"
    else:
        return "Unknown"
