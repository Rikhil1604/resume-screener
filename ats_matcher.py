import re
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_from_jd(jd_text, top_n=10):
    """Extract top-N keywords from a job description using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_keywords = [word for word, score in sorted_scores[:top_n]]
    return top_keywords

def calculate_ats_match(resume_text, job_title="", jd_keywords=None):
    """Calculate ATS score using job keywords or extracted JD keywords."""
    # Hardcoded keywords as fallback
    job_keywords = {
        "data scientist": ["machine learning", "python", "data", "model", "analysis"],
        "software engineer": ["java", "python", "api", "sql", "development"],
        "hr": ["recruiting", "payroll", "training", "compliance"],
        "designer": ["adobe", "figma", "ui", "ux", "illustrator"],
    }

    resume_lower = resume_text.lower()

    if jd_keywords:
        keywords = jd_keywords
    else:
        title = job_title.lower()
        keywords = job_keywords.get(title, [])

    matched = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", resume_lower)]
    score = int((len(matched) / len(keywords)) * 100) if keywords else 0
    missing = list(set(keywords) - set(matched))

    return score, missing

def suggest_similar_roles(resume_text):
    """Suggest possible roles based on keywords found in resume."""
    roles = {
        "Data Scientist": ["machine learning", "pandas", "dataframe", "statistics"],
        "Software Engineer": ["java", "c++", "backend", "frontend", "api", "django", "flask"],
        "HR": ["employee relations", "recruiting", "onboarding", "training"],
        "Designer": ["photoshop", "illustrator", "figma", "ux", "ui"],
        "Manager": ["project management", "team lead", "planning", "budgeting"]
    }

    suggestions = []
    resume_lower = resume_text.lower()
    for role, keywords in roles.items():
        for kw in keywords:
            if kw in resume_lower:
                suggestions.append(role)
                break

    return list(set(suggestions))
