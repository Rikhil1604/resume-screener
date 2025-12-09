A simple but powerful AI-driven Resume Screener that uses LLMs + ML + ATS logic to give instant, structured feedback on any resume. Upload a PDF → get a detailed score, improvement suggestions, missing keywords, and a clean downloadable report.

This project shows hands-on experience with LLM integration, ML pipelines, NLP, Streamlit apps, and deploying real usable tools.

🚀 What It Does

LLM Feedback (Cohere Chat API)
Generates a score out of 100, strengths, weaknesses, and personalized suggestions.

ML Resume Category Prediction
Classifies the resume using a trained scikit-learn model.

ATS Keyword Matching
Finds missing keywords based on job description or role.

Resume Freshness Detection
Estimates the last update year from the resume text.

Suggests Similar Roles
Based on content patterns in the resume.

Downloadable PDF Report
Clean, ready-to-share summary of the analysis.

🛠️ Tech Stack
-Python, Streamlit
-Cohere Chat API (command-a-03-2025)
-scikit-learn, joblib
-regex/NLP utilities

HTML → PDF generation

Streamlit Cloud Deployment

🎯 Why I Built This

I wanted a real, practical AI project that:
-Combines ML + LLMs in a clean architecture
-Is actually useful for people
-Can be deployed and shared publicly
-Shows I understand how to integrate multiple AI components together

This is a complete, polished end-to-end product, not just a notebook.

💻 Running Locally
pip install -r requirements.txt
streamlit run app.py


Add your Cohere key in .streamlit/secrets.toml:
[cohere]
api_key = "YOUR_KEY"

📂 Project Layout
app.py
scorer.py
resume_parser.py
ats_matcher.py
pdf_generator.py
models/

🔮 Future Plans
-Add embeddings + vector search (RAG)
-Smarter ATS scoring
-JD–resume semantic similarity
-UI polish + better visual summaries
