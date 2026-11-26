import re, io
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

ROLE_SKILLS = {
 "Data Analyst": ["Python","SQL","Excel","Pandas","Power BI","Tableau","Statistics","NumPy"],
 "Machine Learning Engineer": ["Python","Machine Learning","scikit-learn","Pandas","NumPy","TensorFlow","PyTorch","FastAPI","Docker","SQL"],
 "AI Engineer": ["Python","Deep Learning","LLM","RAG","APIs","PyTorch","TensorFlow","NLP","Transformers","Docker"],
 "NLP Engineer": ["Python","NLP","Transformers","Hugging Face","spaCy","Machine Learning","PyTorch","LLM","Text Classification"],
 "Computer Vision Engineer": ["Python","OpenCV","CNN","YOLO","Deep Learning","PyTorch","TensorFlow","Image Processing"],
 "Python Developer": ["Python","OOP","Git","REST API","FastAPI","Django","Flask","SQL","Testing"],
 "Data Engineer": ["Python","SQL","ETL","Spark","Hadoop","Airflow","AWS","Docker","Data Warehousing"],
 "Web Developer": ["HTML","CSS","JavaScript","React","Node.js","REST API","Git","SQL"],
}
SKILLS = sorted(set(s for values in ROLE_SKILLS.values() for s in values) | {"Java","C","C++","JavaScript","MongoDB","MySQL","PostgreSQL","AWS","Azure","Linux","Kubernetes","GitHub","Jupyter","Matplotlib","Seaborn","Keras","Scikit-learn","Flask","Django","HTML","CSS","React","Node.js","Computer Networks","DBMS","Data Structures","Algorithms","Agile","Communication","Problem Solving"}, key=len, reverse=True)
ROADMAP = {
 "Python":"Practice Python fundamentals, functions, OOP and problem-solving.", "SQL":"Learn SELECT, JOIN, GROUP BY, subqueries and window functions.", "Pandas":"Practice data cleaning, filtering, grouping and merging datasets.", "Power BI":"Build dashboards using Power BI and learn data modelling.", "Machine Learning":"Study supervised learning, evaluation metrics and model validation.", "scikit-learn":"Practice preprocessing, pipelines, model training and evaluation.", "FastAPI":"Build and test REST endpoints for a small ML model.", "Docker":"Containerize a Python application and learn images and containers.", "NLP":"Learn text preprocessing, tokenization, embeddings and text classification.", "Transformers":"Study attention and use a pretrained transformer for a task.", "Deep Learning":"Learn neural networks, backpropagation and model training.", "LLM":"Learn prompt design, tokenization and LLM application basics.", "RAG":"Build a retrieval pipeline with embeddings and a vector index.", "OpenCV":"Practice image reading, filtering, contours and object detection.", "YOLO":"Train or run a pretrained object detector on sample images.", "AWS":"Learn core cloud services and deploy a small application.", "Spark":"Practice distributed data processing with PySpark.", "React":"Build reusable components and connect them to APIs.", "Git":"Practice branching, commits, pull requests and resolving conflicts.",
}
def extract_text(file):
    name=file.name.lower(); data=file.getvalue()
    if name.endswith('.pdf'):
        reader=PdfReader(io.BytesIO(data)); return '\n'.join((p.extract_text() or '') for p in reader.pages)
    if name.endswith('.docx'):
        doc=Document(io.BytesIO(data)); return '\n'.join(p.text for p in doc.paragraphs)
    return ''

def clean_text(text):
    text=text.lower()
    text=re.sub(r'\s+',' ',text)
    return text.strip()

def find_skills(text):
    t=clean_text(text); found=[]
    for skill in SKILLS:
        pattern=r'(?<![a-z0-9])'+re.escape(skill.lower())+r'(?![a-z0-9])'
        if re.search(pattern,t): found.append(skill)
    return sorted(set(found),key=str.lower)

def score_roles(text):
    role_texts=[' '.join(v) for v in ROLE_SKILLS.values()]
    docs=[text]+role_texts
    tfidf=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
    matrix=tfidf.fit_transform(docs)
    scores=cosine_similarity(matrix[0:1],matrix[1:]).flatten()
    return sorted(zip(ROLE_SKILLS.keys(),scores),key=lambda x:x[1],reverse=True)

st.title("📄 AI Resume Analyzer & Job Recommendation System")
st.caption("Skill-based educational guidance. Match scores are estimates, not hiring decisions.")
with st.sidebar:
    st.header("Upload Resume")
    uploaded=st.file_uploader("Choose PDF or DOCX",type=["pdf","docx"])
    st.info("Resume text is processed in this session and is not intentionally saved to disk.")
if uploaded:
    raw=extract_text(uploaded)
    if not raw.strip():
        st.error("No selectable text found. Try a text-based PDF or DOCX (scanned PDFs need OCR).")
    else:
        text=clean_text(raw); found=find_skills(raw); scores=score_roles(text)
        st.success(f"Processed: {uploaded.name}")
        c1,c2,c3=st.columns(3); c1.metric("Skills detected",len(found)); c2.metric("Roles compared",len(ROLE_SKILLS)); c3.metric("Top match",f"{scores[0][1]*100:.1f}%")
        st.subheader("Skills found in your resume")
        st.write(', '.join(found) if found else "No skills from the current dictionary were detected. Add relevant skill terms to your resume if accurate.")
        st.subheader("Top 3 recommended roles")
        chart=pd.DataFrame({"Job Role":[x[0] for x in scores[:3]],"Match Score (%)":[round(x[1]*100,1) for x in scores[:3]]}).set_index("Job Role")
        st.bar_chart(chart)
        st.dataframe(pd.DataFrame({"Role":[x[0] for x in scores],"TF-IDF cosine match (%)":[round(x[1]*100,1) for x in scores]}),use_container_width=True,hide_index=True)
        role=st.selectbox("Select a target role for skill-gap analysis",list(ROLE_SKILLS.keys()),index=list(ROLE_SKILLS.keys()).index(scores[0][0]))
        required=ROLE_SKILLS[role]; missing=[s for s in required if s.lower() not in text]
        st.subheader(f"Skill gap: {role}")
        a,b=st.columns(2); a.markdown("**Skills found**"); a.write([s for s in required if s.lower() in text] or "None detected")
        b.markdown("**Skills not detected**"); b.write(missing or "All listed skills detected")
        st.subheader("Suggested learning roadmap")
        for i,skill in enumerate(missing[:8],1): st.markdown(f"**Step {i}: {skill}** — {ROADMAP.get(skill, 'Study the fundamentals and complete a small hands-on project.')}")
        report=f"Resume analysis: {uploaded.name}\nDetected skills: {', '.join(found)}\n\nRole scores:\n"+'\n'.join(f'{r}: {s*100:.1f}%' for r,s in scores)+f"\n\nTarget role: {role}\nMissing skills: {', '.join(missing)}"
        st.download_button("Download analysis as TXT",report,file_name="resume_analysis.txt")
else:
    st.markdown("Upload a resume to view detected skills, role-match estimates, recommendations and a learning roadmap.")
st.caption("Limitations: keyword dictionary and TF-IDF similarity can miss equivalent skills or context. Review results manually; this tool should not be used to automatically hire or reject anyone.")
