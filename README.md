# Explainable-Fake-News-Detection
## Overview

This project focuses on building a Fake News Detection System using Natural Language Processing (NLP) and Transformer-based models. The goal is to accurately classify news articles as real or fake while providing human-readable explanations for each prediction. Using BERT for classification and T5 for text-based explanations, the project combines the power of contextual understanding and generative modeling to enhance both accuracy and transparency in fake news detection. The system is trained and evaluated on the LIAR dataset, which contains labeled political statements with supporting metadata.

## Problem Statement

The rise of fake news has become one of the most pressing challenges in the digital information age. With social media and online platforms allowing rapid spreading of unverified information, distinguishing between factual and fabricated news has grown increasingly difficult. This project aims to develop a machine learning model that can automatically detect fake news using Natural Language Processing (NLP) techniques. The goal is to not only classify news articles as fake or real but also provide explanations for the model’s predictions to enhance transparency and trust.

## Data Overview

The dataset used in this project is the LIAR dataset, which contains over 12,000 labeled statements collected from various political debates, social media platforms, and fact-checking organizations. Each entry includes the statement text, a truthfulness label (ranging from pants-on-fire to true), and accompanying metadata such as the speaker, subject, and context. For this project, the task is simplified into a binary classification problem - categorizing statements as either fake or real. The dataset is divided into training, validation, and test subsets to ensure fair evaluation and model generalization.

## Methodologies





1. Data Preprocessing & Cleaning – The text data is cleaned to remove unwanted characters, punctuation, and stopwords. Tokenization and lowercasing are applied to prepare the text for modeling. The metadata is also explored for potential auxiliary features that could improve prediction accuracy.

2. Exploratory Data Analysis (EDA) – Before modeling, EDA is performed to uncover trends in fake versus real statements, analyze label balance, and identify key words or phrases frequently associated with misinformation.

3. Model Training & Fine-Tuning – A two-stage approach is adopted:

    Stage 1 (BERT): Fine-tuned for binary classification to predict fake or real.

    Stage 2 (T5): Fine-tuned for text generation to produce short, natural language explanations for the model’s predictions.

4. Explainability & Evaluation – Explainability & Evaluation – Explainability is achieved through both T5-generated explanations and interpretability frameworks like LIME or SHAP, which highlight the most influential text features. Performance is evaluated using accuracy, F1-score, and confusion matrix analysis.

## Model Architecture

This project implements two transformer models combining the strengths of BERT and T5.
### Transformers
-Transformer model is a type of neural network architecture that excels at processing sequential data. Transformers are based on a mechanism called self-attention, which allows the model to weigh the importance of different words in a sequence relative to each other regardless of their position in the text.

#### Transformer architecture
Transformer has 4 main parts:
1. Tokenization - Tokenization is the most basic step. It consists of a large dataset of tokens, including all the words, punctuation signs, etc. The tokenization step takes every word, prefix, suffix, and punctuation signs, and sends them to a known token from the library.
2. Embedding - After tokenizing the input, words are converted into numerical representations called embeddings. Each piece of text is mapped to a vector of numbers, where similar texts have similar vectors (their values are close component by component), and different texts have distinct vectors.
3. Positional encoding - Positional encoding consists of adding a sequence of predefined vectors to the embedding vectors of the words. This ensures we get a unique vector for every sentence, and sentences with the same words in different order will be assigned different vectors.
4. Transformer blocks - is formed by two main components:
   * The attention component.
   * The feedforward component
   
<img width="747" height="444" alt="image" src="https://github.com/user-attachments/assets/cdd8415f-1dc0-4605-a80b-b25c1cb7a728" />


### Stage 1: BERT (Bidirectional Encoder Representations from Transformers)
BERT is used for fake news classification. It processes text bidirectionally, capturing context from both directions in a sentence. This allows it to understand complex linguistic cues, sarcasm, and context-dependent meanings — all of which are essential in identifying fake news. The model’s output is a binary prediction indicating whether the input statement is fake or real.

### Stage 2: T5 (Text-to-Text Transfer Transformer)
Once BERT classifies the news, T5 generates a human-readable explanation for the classification. The T5 model is a transformer based architecture that simplifies NLP task by converting them into a common text to text format.

#### The Architecture of T5
The T5 model builds upon the transformer architecture with key components like.
##### Encoder-Decoder design
  * Encoder - Processes the input text and creates a meaningful representation.
  * Decoder - Generates the output text based on the encoder’s representation.

##### Attention Mechanisms
 * T5 employs self-attention in the encoder to focus on relevant parts of the input.
 * The decoder uses both self-attention and encoder decoder attention for generating context-aware outputs.

##### Text to text paradigm
Every task is reformulated as text input - text output, ensuring uniformity across applications.
<img width="550" height="349" alt="image" src="https://github.com/user-attachments/assets/3783f045-2bf8-467b-921b-32987ce8254c" />

