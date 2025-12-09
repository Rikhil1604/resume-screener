import os
import streamlit as st
import joblib
import traceback

from scorer import score_resume
from resume_parser import extract_text_from_pdf, estimate_resume_freshness
from ats_matcher import calculate_ats_match, suggest_similar_roles, extract_keywords_from_jd
from pdf_generator import convert_html_to_pdf

# Setup 
st.set_page_config(page_title="AI Resume Screener", layout="centered")

# Dark Theme Style
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

# Load Cohere API Key (safe)
cohere_api_key = None
try:
    cohere_api_key = st.secrets["cohere"]["api_key"]
except Exception:
    # optional: attempt to read environment variable if you set it that way
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        st.warning("⚠️ Cohere API key not found. Add it to Streamlit Secrets as st.secrets['cohere']['api_key'] or set COHERE_API_KEY env var.")

# Load ML models safely (fail gracefully on cloud)
pipeline = None
label_encoder = None
try:
    pipeline = joblib.load("models/svm_pipeline.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
except FileNotFoundError:
    st.info("⚠️ ML models not found in /models. Category prediction will be disabled until models are added.")
except Exception as e:
    st.warning(f"⚠️ Error loading ML models: {e}")

# Sidebar Controls 
st.sidebar.title("🎛️ Options")
job_role = st.sidebar.text_input("🎯 Target Job Title", value="Data Scientist")
detail_level = st.sidebar.radio("✍️ Feedback Style", ["Brief", "Detailed"], horizontal=True)
uploaded_file = st.sidebar.file_uploader("📎 Upload Resume (PDF only)", type="pdf")
generate_button = st.sidebar.button("🚀 Generate Analysis", type="primary")

# JD Input
st.subheader("📝 Paste Job Description (Optional)")
jd_text = st.text_area("Helps tailor feedback and improve keyword matching.", height=200)

# Title 
st.title("📄 AI Resume Screener")
st.markdown("""
Upload your resume and click the Generate Analysis button to receive:
- ✅ LLM-powered job-fit feedback
- 🧠 Resume category prediction (ML)
- 📊 ATS score based on JD
- 📅 Resume freshness estimate
- 💡 Suggested similar roles
- 📌 JD keyword extraction  
""")

# ML Category Predictor (safe)
def predict_category(resume_text):
    if pipeline is None or label_encoder is None:
        return "N/A (models missing)"
    try:
        return label_encoder.inverse_transform(pipeline.predict([resume_text]))[0]
    except Exception:
        return "N/A (prediction failed)"

# Main Logic 
if uploaded_file and generate_button:
    with st.spinner("🔎 Analyzing your resume..."):
        feedback = None
        try:
            resume_text = extract_text_from_pdf(uploaded_file)

            # JD Keywords - fallback to job_role if JD not provided
            jd_keywords = extract_keywords_from_jd(jd_text) if jd_text else extract_keywords_from_jd(job_role)

            # LLM Feedback
            if not cohere_api_key:
                st.error("❌ Missing Cohere API key. Add it to Streamlit secrets and redeploy.")
            else:
                feedback = score_resume(
                    resume_text,
                    job_title=job_role,
                    api_key=cohere_api_key,
                    mode=detail_level.lower(),
                    job_description=jd_text
                )
                st.success("✅ LLM Feedback Generated")
                st.markdown(feedback)

            # Resume Category (ML)
            category = predict_category(resume_text)
            st.subheader("🧠 Predicted Resume Category")
            st.markdown(f"**{category}**")

            # ATS Match
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

            # Recency
            freshness = estimate_resume_freshness(resume_text)
            st.subheader("📅 Resume Freshness Estimate")
            st.markdown(f"🗓️ Last update appears to be from: **{freshness}**")

            # Role Suggestions
            similar_roles = suggest_similar_roles(resume_text)
            st.subheader("💡 Suggested Job Roles")
            if similar_roles:
                st.markdown(", ".join(similar_roles))
            else:
                st.markdown("_No strong matches found._")

            # JD Keyword Display
            if jd_keywords:
                st.subheader("📌 Extracted JD Keywords")
                st.markdown(", ".join(jd_keywords))

            # PDF Feedback Download — ensure feedback fallback
            st.download_button(
                label="📥 Download Feedback as PDF",
                data=convert_html_to_pdf(
                    feedback_text=feedback if feedback else "No feedback generated.",
                    job_title=job_role,
                    category=category,
                    ats_score=final_score,
                    freshness=freshness,
                    jd_keywords=jd_keywords
                ),
                file_name="resume_feedback.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            # Inspect for HTTP-like status codes (covers Cohere SDK or requests)
            status = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
            if status == 401:
                st.error("❌ Invalid Cohere API key or unauthorized access.")
            elif status == 429:
                st.error("❌ Rate limit exceeded. Try again later.")
            elif status == 500:
                st.error("❌ Server error. Please try again later.")
            else:
                st.error(f"❌ Unexpected error: {e}")
                # For debugging only (remove in production)
                st.text("Debug info (traceback):")
                st.text(traceback.format_exc())

elif uploaded_file and not generate_button:
    st.info("📄 Resume uploaded. Click 'Generate Analysis' in the sidebar to analyze your resume.")
