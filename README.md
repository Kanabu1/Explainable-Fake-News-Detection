# Explainable-Fake-News-Detection
## Problem Statement

The rise of fake news has become one of the most pressing challenges in the digital information age. With social media and online platforms allowing rapid spreading of unverified information, distinguishing between factual and fabricated news has grown increasingly difficult. This project aims to develop a machine learning model that can automatically detect fake news using Natural Language Processing (NLP) techniques. The goal is to not only classify news articles as fake or real but also provide explanations for the model’s predictions to enhance transparency and trust.

## Data Overview

The dataset used in this project is the LIAR dataset, which contains over 12,000 labeled statements collected from various political debates, social media platforms, and fact-checking organizations. Each entry includes the statement text, a truthfulness label (ranging from pants-on-fire to true), and accompanying metadata such as the speaker, subject, and context. For this project, the task is simplified into a binary classification problem — categorizing statements as either fake or real. The dataset is divided into training, validation, and test subsets to ensure fair evaluation and model generalization.

## Methodologies

This project follows a systematic and research-driven methodology comprising five key stages:

Data Collection & Exploration – The project uses the LIAR dataset, which contains labeled short political statements alongside metadata such as speaker, context, and truth labels. The dataset is inspected for completeness, inconsistencies, and class distribution to ensure readiness for modeling.

Data Preprocessing & Cleaning – The text data is cleaned to remove unwanted characters, punctuation, and stopwords. Tokenization and lowercasing are applied to prepare the text for modeling. The metadata is also explored for potential auxiliary features that could improve prediction accuracy.

Exploratory Data Analysis (EDA) – Before modeling, EDA is performed to uncover trends in fake versus real statements, analyze label balance, and identify key words or phrases frequently associated with misinformation.


Model Training & Fine-Tuning – A two-stage approach is adopted:

Stage 1 (BERT): Fine-tuned for binary classification to predict fake or real.

Stage 2 (T5): Fine-tuned for text generation to produce short, natural language explanations for the model’s predictions.

Explainability & Evaluation – Explainability & Evaluation – Explainability is achieved through both T5-generated explanations and interpretability frameworks like LIME or SHAP, which highlight the most influential text features. Performance is evaluated using accuracy, F1-score, and confusion matrix analysis.

## Model Architecture

This project implements two transformer models combining the strengths of BERT and T5.

Stage 1: BERT (Bidirectional Encoder Representations from Transformers)
BERT is used for fake news classification. It processes text bidirectionally, capturing context from both directions in a sentence. This allows it to understand complex linguistic cues, sarcasm, and context-dependent meanings — all of which are essential in identifying fake news. The model’s output is a binary prediction indicating whether the input statement is fake or real.

Stage 2: T5 (Text-to-Text Transfer Transformer)
Once BERT classifies the news, T5 generates a human-readable explanation for the classification. As a text-to-text model, T5 can convert structured model outputs into natural language reasoning
