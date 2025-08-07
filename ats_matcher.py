import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter

def extract_keywords_from_jd(jd_text, top_n=15):
    """Extract top-N keywords from a job description using TF-IDF with n-grams."""
    if not jd_text or len(jd_text.strip()) < 10:
        return []
        
    # Use n-grams to capture multi-word terms
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include single words and bigrams
        max_df=0.85,         # Ignore terms that appear in >85% of docs
        min_df=1             # Only consider terms that appear at least once
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores from the matrix
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Filter out single-letter words and pure numbers
        filtered_keywords = []
        for word, score in sorted_scores:
            if len(word) > 1 and not word.isdigit() and score > 0:
                filtered_keywords.append((word, score))
        
        top_keywords = [word for word, score in filtered_keywords[:top_n]]
        return top_keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def calculate_ats_match(resume_text, job_title="", jd_keywords=None):
    """Calculate ATS score using job keywords or extracted JD keywords with fuzzy matching."""
    # Expanded hardcoded keywords as fallback
    job_keywords = {
        "data scientist": [
            "machine learning", "python", "data analysis", "statistics", "sql", 
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", 
            "data visualization", "r", "big data", "data mining", "nlp",
            "deep learning", "ai", "predictive modeling", "data science"
        ],
        "software engineer": [
            "java", "python", "javascript", "c++", "c#", "api", "sql", "nosql",
            "rest", "git", "cloud", "aws", "azure", "docker", "kubernetes",
            "microservices", "agile", "devops", "ci/cd", "web development"
        ],
        "hr": [
            "recruiting", "payroll", "training", "compliance", "onboarding", 
            "employee relations", "benefits", "compensation", "hr information systems",
            "talent acquisition", "workforce planning", "performance management"
        ],
        "designer": [
            "adobe", "figma", "ui", "ux", "illustrator", "photoshop", "indesign",
            "sketch", "typography", "responsive design", "wireframing", "prototyping",
            "user research", "visual design", "interaction design"
        ],
        "manager": [
            "project management", "team leadership", "strategic planning", "budgeting",
            "stakeholder management", "risk management", "resource allocation",
            "performance reviews", "process improvement", "kpis", "leadership"
        ]
    }

    # Normalize and clean text
    resume_lower = resume_text.lower()
    
    # Handle edge cases
    if not resume_text or (not jd_keywords and not job_title):
        return 0, []
    
    # Get keywords based on input
    if jd_keywords and len(jd_keywords) > 0:
        keywords = jd_keywords
    else:
        # Try to match the job title to our dictionary, with fallback options
        title = job_title.lower()
        keywords = []
        
        # Try exact match first
        if title in job_keywords:
            keywords = job_keywords[title]
        else:
            # Try partial matches
            for key in job_keywords:
                if key in title or title in key:
                    keywords = job_keywords[key]
                    break
        
        # If still no match, use data scientist as default if "data" in title
        if not keywords and "data" in title:
            keywords = job_keywords["data scientist"]
        # Otherwise use software engineer as a generic fallback
        elif not keywords:
            keywords = job_keywords["software engineer"]
    
    # Different matching algorithms
    exact_matches = []
    fuzzy_matches = []
    
    for kw in keywords:
        # Exact match (word boundaries)
        if re.search(rf"\b{re.escape(kw)}\b", resume_lower):
            exact_matches.append(kw)
        # Fuzzy match (within words, no boundaries)
        elif kw in resume_lower:
            fuzzy_matches.append(kw)
    
    # Calculate scores with weighting
    matched = exact_matches + fuzzy_matches
    exact_match_count = len(exact_matches)
    fuzzy_match_count = len(fuzzy_matches)
    
    # Weight exact matches more heavily
    weighted_match_score = (exact_match_count * 1.0) + (fuzzy_match_count * 0.5)
    
    # Calculate score as percentage with ceiling of 100
    keyword_count = len(keywords)
    if keyword_count > 0:
        raw_score = (weighted_match_score / keyword_count) * 100
        # Apply a curve to make scores more reasonable
        # Square root curve gives more reasonable scores for partial matches
        score = min(100, int(np.sqrt(raw_score) * 10))
    else:
        score = 0
    
    # Find missing keywords
    missing = list(set(keywords) - set(matched))
    
    return score, missing

def suggest_similar_roles(resume_text):
    """Suggest possible roles based on keywords found in resume with ranking."""
    if not resume_text:
        return []
        
    # Expanded role definitions with more keywords
    roles = {
        "Data Scientist": [
            "machine learning", "pandas", "numpy", "statistics", "python", 
            "data analysis", "sql", "r", "tensorflow", "pytorch", "scikit-learn",
            "big data", "data mining", "visualization", "predictive modeling",
            "regression", "classification", "clustering", "nlp", "deep learning"
        ],
        "Data Engineer": [
            "etl", "data pipeline", "sql", "nosql", "hadoop", "spark", "airflow",
            "kafka", "database", "data warehouse", "data lake", "aws", "azure",
            "gcp", "python", "scala", "java", "distributed systems"
        ],
        "Machine Learning Engineer": [
            "machine learning", "deep learning", "neural networks", "tensorflow",
            "pytorch", "keras", "model deployment", "mlops", "feature engineering",
            "hyperparameter tuning", "python", "distributed training"
        ],
        "Software Engineer": [
            "java", "c++", "c#", "javascript", "python", "backend", "frontend", 
            "full stack", "api", "rest", "microservices", "django", "flask", "node.js",
            "react", "angular", "vue", "docker", "kubernetes", "aws", "git"
        ],
        "Frontend Developer": [
            "javascript", "html", "css", "react", "angular", "vue", "webpack",
            "responsive design", "ui", "ux", "typescript", "sass", "less", "dom"
        ],
        "Backend Developer": [
            "java", "python", "c#", "node.js", "api", "rest", "graphql", "database",
            "sql", "nosql", "microservices", "django", "flask", "spring", "express"
        ],
        "DevOps Engineer": [
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
            "jenkins", "ci/cd", "automation", "monitoring", "linux", "unix", "shell"
        ],
        "HR Specialist": [
            "recruiting", "talent acquisition", "employee relations", "payroll",
            "benefits", "compliance", "onboarding", "training", "hr information systems",
            "performance management", "compensation"
        ],
        "UX/UI Designer": [
            "user experience", "user interface", "figma", "sketch", "adobe xd", 
            "wireframing", "prototyping", "user research", "usability testing",
            "interaction design", "visual design", "responsive design"
        ],
        "Product Manager": [
            "product development", "roadmap", "agile", "scrum", "user stories",
            "market research", "stakeholder management", "requirements", "backlog",
            "kpis", "metrics", "product strategy", "a/b testing"
        ],
        "Project Manager": [
            "project management", "agile", "scrum", "waterfall", "pmbok", "pmp",
            "budgeting", "resource allocation", "gantt", "risk management",
            "stakeholder management", "project planning", "team leadership"
        ],
        "Business Analyst": [
            "requirements gathering", "data analysis", "sql", "business intelligence",
            "process improvement", "user stories", "stakeholder management",
            "excel", "visualization", "reporting", "documentation"
        ]
    }

    # Preprocess resume text
    resume_lower = resume_text.lower()
    
    # Count matching keywords for each role
    role_scores = Counter()
    
    for role, keywords in roles.items():
        for kw in keywords:
            if kw in resume_lower:
                # Score is increased by 1 for each matching keyword
                role_scores[role] += 1
    
    # Calculate match percentage for each role
    role_percentages = {}
    for role, score in role_scores.items():
        total_keywords = len(roles[role])
        percentage = (score / total_keywords) * 100
        role_percentages[role] = percentage
    
    # Get top roles (roles with at least 20% match)
    suggested_roles = [role for role, percentage in role_percentages.items() 
                       if percentage >= 20]
    
    # Sort by match percentage
    suggested_roles.sort(key=lambda role: role_percentages[role], reverse=True)
    
    # Return top 3 roles max
    return suggested_roles[:3]