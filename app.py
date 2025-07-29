import streamlit as st
import requests
import joblib
from scorer import score_resume
from resume_parser import extract_text_from_pdf, estimate_resume_freshness
from ats_matcher import calculate_ats_match, suggest_similar_roles, extract_keywords_from_jd
from pdf_generator import convert_html_to_pdf

# ---- Setup ----
st.set_page_config(page_title="AI Resume Screener", layout="centered")

# 🌙 Custom Dark Theme Styles
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #00BFFF;
        }
        .stTextInput > div > input,
        .stTextArea > div > textarea {
            background-color: #1e1e1e;
            color: #E0E0E0;
            border: 1px solid #00BFFF;
        }
        .stButton > button {
            background-color: #00BFFF;
            color: white;
        }
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }
            h1 {
                font-size: 1.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ---- Load API Key and Models ----
cohere_api_key = st.secrets["cohere"]["api_key"]
pipeline = joblib.load("models/svm_pipeline.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---- Sidebar Controls ----
st.sidebar.title("🎛️ Options")
job_role = st.sidebar.text_input("🎯 Target Job Title", value="Data Scientist")
detail_level = st.sidebar.radio("✍️ Feedback Style", ["Brief", "Detailed"], horizontal=True)
uploaded_file = st.sidebar.file_uploader("📎 Upload Resume (PDF only)", type="pdf")

# ---- Title & Intro ----
st.title("📄 AI Resume Screener")
st.markdown("""
Upload your resume to receive:
- ✅ LLM-powered job-fit feedback
- 🧠 Resume category prediction (ML)
- 📊 ATS score based on JD
- 📅 Resume freshness estimate
- 💡 Suggested similar roles
- 📌 JD keyword extraction  
""")

# ---- Prediction Function ----
def predict_category(resume_text):
    return label_encoder.inverse_transform(pipeline.predict([resume_text]))[0]

# ---- Main Logic ----
if uploaded_file:
    with st.spinner("🔎 Analyzing your resume..."):
        try:
            # 1. Extract text
            resume_text = extract_text_from_pdf(uploaded_file)

            # 2. JD Keywords
            jd_keywords = extract_keywords_from_jd(job_role) if job_role else []

            # 3. LLM Resume Feedback
            feedback = score_resume(
                resume_text,
                job_title=job_role,
                api_key=cohere_api_key,
                mode=detail_level.lower()
            )
            st.success("✅ LLM Feedback Generated")
            st.markdown(feedback)

            # 4. ML Resume Category
            category = predict_category(resume_text)
            st.subheader("🧠 Predicted Resume Category")
            st.markdown(f"**{category}**")

            # 5. ATS Match Score
            st.subheader("📊 ATS Match Score")

            hardcoded_score, missing = calculate_ats_match(resume_text, job_role)

            jd_score = 0
            jd_missing = []
            if jd_keywords:
                jd_matches = [kw for kw in jd_keywords if kw.lower() in resume_text.lower()]
                jd_score = int((len(jd_matches) / len(jd_keywords)) * 100) if jd_keywords else 0
                jd_missing = list(set(jd_keywords) - set(jd_matches))

            final_score = max(hardcoded_score, jd_score)
            final_missing = list(set(missing + jd_missing))

            st.markdown(f"**Score:** {final_score}/100")
            if final_missing:
                st.markdown("**🔻 Missing Keywords:**")
                st.markdown(", ".join(final_missing))
            else:
                st.markdown("_Your resume contains all relevant keywords!_")

            # 6. Resume Freshness
            freshness = estimate_resume_freshness(resume_text)
            st.subheader("📅 Resume Freshness Estimate")
            st.markdown(f"🗓️ Last update appears to be from: **{freshness}**")

            # 7. Similar Roles
            similar_roles = suggest_similar_roles(resume_text)
            st.subheader("💡 Suggested Job Roles")
            if similar_roles:
                st.markdown(", ".join(similar_roles))
            else:
                st.markdown("_No strong matches found._")

            # 8. JD Keywords Display
            if jd_keywords:
                st.subheader("📌 Extracted JD Keywords")
                st.markdown(", ".join(jd_keywords))

            # 9. Download PDF
            st.download_button(
                label="📥 Download Feedback as PDF",
                data=convert_html_to_pdf(feedback),
                file_name="resume_feedback.pdf",
                mime="application/pdf"
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("❌ Invalid Cohere API key.")
            elif e.response.status_code == 429:
                st.error("❌ Rate limit exceeded. Try again later.")
            elif e.response.status_code == 500:
                st.error("❌ Server error. Please try again later.")
            else:
                st.error(f"❌ HTTP error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
