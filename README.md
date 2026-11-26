# AI Resume Analyzer and Job Recommendation System

## Features
- PDF/DOCX resume upload and text extraction
- Text normalization and skill dictionary (30+ job-related skills)
- TF-IDF + cosine similarity ranking across 8 job roles
- Top-three role recommendations, skill gaps and a learning roadmap
- Streamlit UI and downloadable text report

## Run locally
1. Install Python 3.10+ and open a terminal in this folder.
2. Create/activate a virtual environment (commands in the main bundle README).
3. Install: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Method
Resume text and each role's required-skill text are vectorized with TF-IDF; cosine similarity gives a relative match estimate. Skill gap is determined from a curated keyword list. Scores are not probabilities or hiring decisions.

## Limitations / responsible use
Keyword matching may miss synonyms or skills described indirectly. Use only job-related information; do not score protected attributes. Do not treat missing keywords as proof of inability. Resume data is processed in memory by this app; avoid uploading confidential resumes to public deployments.
