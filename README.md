# AutoEDA Performer

AI-Powered Automated Exploratory Data Analysis and Data Preprocessing System.

AutoEDA Performer transforms raw datasets into machine-learning-ready data through intelligent profiling, automated preprocessing, and AI-driven recommendations. Built using FastAPI, LangGraph, Ollama, and modern data science tools.

---

## Features

* Dataset Profiling
* Missing Value Detection and Handling
* Duplicate Record Identification
* Outlier Detection
* Automated Data Cleaning
* Feature Engineering Recommendations
* Encoding and Scaling Suggestions
* LLM-Powered Preprocessing Planning
* ML Readiness Assessment
* Processed Dataset Export
* Data Quality Reports

---

## Tech Stack

### Backend

* Python
* FastAPI
* LangGraph

### Data Processing

* Pandas
* NumPy
* Scikit-Learn

### AI & LLM

* Ollama
* Llama 3

### Frontend

* HTML
* CSS
* JavaScript

---

## Workflow

```text
Upload Dataset
    ↓
Dataset Profiling
    ↓
AI Preprocessing Planner
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
ML Readiness Evaluation
    ↓
Generate Reports
    ↓
Download Processed Dataset
```

---

## Project Structure

```text
AutoEDA-Performer/
│
├── backend/
│   ├── api/
│   ├── services/
│   ├── workflow/
│   ├── models/
│   └── main.py
│
├── frontend/
│   ├── css/
│   ├── js/
│   └── index.html
│
├── uploads/
├── outputs/
├── sample_data/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/your-username/AutoEDA-Performer.git
cd AutoEDA-Performer

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Application

### Backend

```bash
uvicorn backend.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

### Frontend

```bash
python -m http.server 5500
```

Frontend URL:

```text
http://localhost:5500
```

---

## Outputs

* Dataset Profile Report
* Missing Value Analysis
* Duplicate Detection Summary
* Outlier Analysis
* Feature Engineering Recommendations
* ML Readiness Score
* Cleaned Dataset Export
* Processed Dataset Export

---

## Future Enhancements

* Automated Feature Selection
* Model Recommendation System
* Data Drift Detection
* PDF Report Generation
* Advanced Visualizations
* Explainable AI Recommendations
* Cloud Deployment Support

---



## Author

**Deepak Vaidhyanathan**
B.Tech Artificial Intelligence and Machine Learning
Sri Shakthi Institute of Engineering and Technology
