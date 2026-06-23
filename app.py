import streamlit as st
import pandas as pd

from utils import extract_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🤖 AI-Powered Resume Ranker")

st.info(
    "Upload resumes and compare them against a Job Description using NLP techniques."
)

st.markdown("""
### Features

✅ PDF Resume Upload  
✅ Resume Text Extraction  
✅ Skill Detection  
✅ TF-IDF Vectorization  
✅ Cosine Similarity Matching  
✅ Candidate Ranking  
✅ CSV Export  
✅ Data Visualization
""")

# ---------------------------------------------------
# Skills Database
# ---------------------------------------------------

skills = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "tensorflow",
    "keras",
    "pandas",
    "numpy",
    "scikit-learn",
    "nlp",
    "data science",
    "power bi",
    "excel",
    "java",
    "aws",
    "docker",
    "flask",
    "streamlit",
    "git",
    "github"
]

# ---------------------------------------------------
# Inputs
# ---------------------------------------------------

job_desc = st.text_area(
    "📋 Paste Job Description"
)

uploaded_files = st.file_uploader(
    "📄 Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(
        f"Total Resumes Uploaded: {len(uploaded_files)}"
    )

# ---------------------------------------------------
# Ranking Function
# ---------------------------------------------------

def rank_resumes(job_desc, resumes):

    documents = [job_desc] + resumes

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    matrix = tfidf.fit_transform(
        documents
    )

    scores = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )

    return scores[0]

# ---------------------------------------------------
# Main Button
# ---------------------------------------------------

if st.button("🚀 Rank Resumes"):

    if not job_desc:
        st.warning(
            "Please enter a Job Description."
        )
        st.stop()

    if not uploaded_files:
        st.warning(
            "Please upload at least one resume."
        )
        st.stop()

    resume_texts = []
    resume_names = []
    resume_skills = []

    for file in uploaded_files:

        text = extract_text(file)

        resume_texts.append(text)

        resume_names.append(file.name)

        found_skills = []

        for skill in skills:

            if skill.lower() in text.lower():

                found_skills.append(skill)

        if found_skills:
            resume_skills.append(
                ", ".join(found_skills)
            )
        else:
            resume_skills.append(
                "No skills detected"
            )

    scores = rank_resumes(
        job_desc,
        resume_texts
    )

    results = pd.DataFrame({

        "Resume": resume_names,

        "Match %": (
            scores * 100
        ).round(2),

        "Skills Found": resume_skills

    })

    results = results.sort_values(
        by="Match %",
        ascending=False
    )

    results = results.reset_index(
        drop=True
    )

    # ---------------------------------------------------
    # Best Candidate
    # ---------------------------------------------------

    top_candidate = results.iloc[0]

    st.success(
        f"""
🏆 Best Candidate: {top_candidate['Resume']}

📊 Match Score: {top_candidate['Match %']}%
"""
    )

    # ---------------------------------------------------
    # Results Table
    # ---------------------------------------------------

    st.subheader("📊 Resume Ranking Results")

    st.dataframe(
        results,
        use_container_width=True
    )

    # ---------------------------------------------------
    # Bar Chart
    # ---------------------------------------------------

    st.subheader(
        "📈 Resume Match Score Visualization"
    )

    chart_data = results.set_index(
        "Resume"
    )["Match %"]

    st.bar_chart(
        chart_data
    )

    # ---------------------------------------------------
    # Download CSV
    # ---------------------------------------------------

    csv = results.to_csv(
        index=False
    )

    st.download_button(
        label="⬇ Download Results CSV",
        data=csv,
        file_name="resume_ranking.csv",
        mime="text/csv"
    )

    # ---------------------------------------------------
    # Skills Section
    # ---------------------------------------------------

    st.subheader(
        "🛠 Skills Detected in Resumes"
    )

    for i in range(len(results)):

        st.markdown(
            f"### 📄 {results.iloc[i]['Resume']}"
        )

        st.write(
            results.iloc[i]["Skills Found"]
        )

        st.write("---")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, NLP, TF-IDF and Cosine Similarity"
)