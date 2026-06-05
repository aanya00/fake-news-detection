# Fake News Detection using NLP and Machine Learning

## Overview

This project is a Fake News Detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP) and Machine Learning techniques.

The application preprocesses news text, converts it into numerical features, and predicts whether the news article is trustworthy.

---

## Features

- Text preprocessing and cleaning
- Stopword removal
- Text vectorization
- Machine Learning classification
- Real-time prediction through Flask web interface
- User-friendly web application

---

## Tech Stack

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### NLP
- NLTK
- TF-IDF Vectorization

### Frontend
- HTML
- CSS

---

## Project Structure

fake-news-detection/

│

├── app.py

├── model/

├── templates/

├── static/

├── dataset/

├── requirements.txt

└── README.md

---

## How It Works

1. User enters a news article.
2. Text is cleaned and preprocessed.
3. TF-IDF converts text into numerical features.
4. Trained machine learning model analyzes the text.
5. System predicts:
   - Real News
   - Fake News

---

## Installation

### Clone Repository

git clone https://github.com/aanya00/fake-news-detection.git

### Move into Project Folder

cd fake-news-detection

### Install Dependencies

pip install -r requirements.txt

### Run Application

python app.py

---

## Future Improvements

- Deep Learning models
- BERT-based classification
- News source credibility analysis
- Multi-language support

---

## Author

Aanya Agrawal

B.Tech Computer Science (Artificial Intelligence)

GitHub: https://github.com/aanya00
