# scorer.py
"""
Cohere resume scoring using the Chat API with the singular `message=` arg.
This matches the Cohere SDK variant installed in your environment.
Model used: command-a-03-2025
"""

import cohere

MODEL_NAME = "command-a-03-2025"   # exact model you've confirmed

def _extract_text_from_response(resp):
    # Newer SDKs expose resp.text
    if getattr(resp, "text", None):
        return resp.text.strip()

    # Chat-style message.content blocks
    msg = getattr(resp, "message", None)
    if msg:
        content = getattr(msg, "content", None)
        if isinstance(content, (list, tuple)):
            parts = []
            for b in content:
                if isinstance(b, dict) and "text" in b:
                    parts.append(b["text"])
                else:
                    t = getattr(b, "text", None)
                    if t:
                        parts.append(t)
            if parts:
                return "\n".join(parts).strip()
        else:
            t = getattr(content, "text", None)
            if t:
                return t.strip()

    # Older generate-like shapes
    gens = getattr(resp, "generations", None)
    if gens:
        try:
            first = gens[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"].strip()
            t = getattr(first, "text", None)
            if t:
                return t.strip()
        except Exception:
            pass

    # Fallback: raw dict
    if isinstance(resp, dict):
        if "text" in resp and isinstance(resp["text"], str):
            return resp["text"].strip()
        gens = resp.get("generations")
        if isinstance(gens, list) and gens and isinstance(gens[0], dict) and "text" in gens[0]:
            return gens[0]["text"].strip()

    return None


def score_resume(resume_text, job_title="", api_key="", mode="brief", job_description=""):
    """
    Calls Cohere Chat API using `message=` (singular). Returns text output.
    """

    if not api_key:
        raise ValueError("Cohere API key is required.")

    client = cohere.Client(api_key)

    # Build context
    if job_description and job_description.strip():
        context = f"for the job described below:\n\"\"\"\n{job_description.strip()}\n\"\"\""
    elif job_title and job_title.strip():
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

    try:
        # NOTE: using singular 'message' because this SDK variant expects it
        response = client.chat(
            model=MODEL_NAME,
            message=prompt,
            temperature=0.3,
            max_tokens=700
        )
    except Exception as e:
        # surface a clear runtime error including provider message
        raise RuntimeError(f"Cohere chat() failed for model '{MODEL_NAME}'. Error: {e}")

    text = _extract_text_from_response(response)
    if text:
        return text

    return f"❌ Error: Could not parse Cohere chat response. Raw response: {repr(response)}"
