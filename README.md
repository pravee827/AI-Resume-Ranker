# 🤖 AI-Powered Resume Ranker

## Overview

The AI-Powered Resume Ranker is a Natural Language Processing (NLP) based web application that automates resume screening and candidate ranking. The system compares uploaded resumes against a job description and ranks candidates according to their relevance.

The application extracts text from PDF resumes, converts the text into numerical vectors using TF-IDF Vectorization, and calculates similarity scores using Cosine Similarity. Based on these scores, candidates are ranked from most suitable to least suitable.

---

## Features

- 📄 Upload multiple PDF resumes
- 📝 Enter a Job Description
- 🤖 Automated Resume Ranking
- 📊 Match Percentage Calculation
- 🛠 Skill Detection
- 🏆 Best Candidate Identification
- 📈 Score Visualization using Charts
- ⬇ Download Results as CSV

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- PDFPlumber
- Natural Language Processing (NLP)

---

## Project Workflow

Resume PDF
↓
Text Extraction (PDFPlumber)
↓
TF-IDF Vectorization
↓
Cosine Similarity
↓
Match Score Calculation
↓
Resume Ranking
↓
Best Candidate Selection

---

## Algorithms Used

### TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF converts textual information into numerical vectors and assigns higher importance to meaningful words.

### Cosine Similarity

Cosine Similarity measures the similarity between resumes and job descriptions by comparing vectorized text representations.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Resume-Ranker.git
