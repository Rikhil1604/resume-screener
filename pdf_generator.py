from xhtml2pdf import pisa
from io import BytesIO

def convert_html_to_pdf(source_html):
    result = BytesIO()
    pisa_status = pisa.CreatePDF(source_html, dest=result)
    if pisa_status.err:
        return None
    return result.getvalue()
