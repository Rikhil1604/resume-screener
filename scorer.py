import requests

def score_resume(resume_text, job_title="", api_key="", mode="brief", job_description=""):
    # Use job description if available, else use title
    if job_description.strip():
        context = f"for the job described below:\n\"\"\"\n{job_description.strip()}\n\"\"\""
    elif job_title.strip():
        context = f"for the job title: **{job_title.strip()}**"
    else:
        context = "for a general corporate role"

    prompt = f"""
You are a helpful AI assistant skilled at evaluating resumes for job applications.

Evaluate the following resume {context}.

Resume:
\"\"\"{resume_text}\"\"\"

Provide the following:
1. A score out of 100 indicating how suitable the resume is for this role.
2. Two strengths of the resume.
3. Two areas of improvement.
4. {'Keep the answer brief (max 80 words).' if mode == 'brief' else 'Give a detailed explanation in around 150-200 words.'}

Start your response with: **Score: <number>/100**
Then list the strengths and areas to improve clearly.
    """

    response = requests.post(
        "https://api.cohere.ai/v1/chat",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "command-r-plus",
            "message": prompt,
            "temperature": 0.3
        }
    )

    response.raise_for_status()
    result = response.json()

    # Return response text cleanly
    try:
        return result["text"].strip()
    except KeyError:
        try:
            return result["generations"][0]["text"].strip()
        except (KeyError, IndexError):
            return "❌ Error: Unexpected response format from Cohere API."
