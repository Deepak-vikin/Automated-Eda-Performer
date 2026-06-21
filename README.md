# Automated-Eda-Performer
Automated Exploratory Data Analysis and intelligent data preprocessing platform with AI-driven preprocessing recommendations.

## Features

- Dataset Profiling
- Missing Value Detection
- Duplicate Detection
- Outlier Analysis
- Automatic Cleaning
- Feature Encoding
- Scaling Recommendations
- ML Readiness Score
- LLM-powered Preprocessing Planning
- Downloadable Processed Dataset

## Tech Stack

- Python
- FastAPI
- Pandas
- NumPy
- Scikit-Learn
- LangGraph
- Ollama
- Llama 3

## Workflow

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
Download Results

## Installation

pip install -r requirements.txt

## Run Backend

uvicorn backend.main:app --reload

## Run Frontend

python -m http.server 5500

## Future Improvements

- Auto Feature Selection
- Auto ML Model Recommendation
- Drift Detection
- PDF Report Generation
