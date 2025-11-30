# Explainable-Fake-News-Detection Using Bert
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

4. Explainability & Evaluation – Explainability & Evaluation – Explainability is achieved through both T5-generated explanations. Performance is evaluated using accuracy, F1-score, and confusion matrix analysis.
## Data Preprocessing and Cleaning
Data quality is critical for detecting subtle linguistic cues in misinformation. Our preprocessing pipeline consists of four distinct stages:

#### 1. Data Integrity Checks



##### Handling Missing Values:

Rows with few missing values  were dropped.

Missing metadata fields (e.g., missing job or state) were filled with a placeholder token ("unknown"). This ensures the Context Injection format remains consistent without crashing the tokenizer.

##### Duplicate Removal:

Checked for duplicate entries.



#### 2. Text Cleaning

Applied cleaning function to standardize the input text before tokenization:

Lowercasing: All text converted to lowercase to maintain consistency for the uncased BERT model.

Noise Removal (Regex):

Removed metadata brackets often found in the raw dataset (e.g., [Video], [Chart]).

Stripped URLs (http://...) and HTML tags to prevent the model from learning shortcuts based on formatting.

Removed words containing numbers (e.g., h1n1) to focus on semantic meaning rather than specific statistics, which BERT cannot verify externally.

#### 3. Feature Engineering

Standard text classifiers only look at the Statement. Hypothesized that the Speaker and Context are vital features. created a composite input string that forces the Transformer to attend to metadata:

Format: [Speaker] ([Party]) stated in [Context]: [Statement]


#### 4. Filtering

The LIAR dataset contains 6 labels. To build a binary detector,  filtered ambiguous data to sharpen the decision boundary:

Removed: Half-True and Barely-True. These labels contain mixed truth/falsehoods that introduce noise into binary training.

Mapped:

False, Pants-Fire $\rightarrow$ FAKE (1)

True, Mostly-True $\rightarrow$ REAL (0)
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
BERT is used for fake news classification. It processes text bidirectionally, capturing context from both directions in a sentence. This allows it to understand complex linguistic cues, sarcasm, and context-dependent meanings all of which are essential in identifying fake news. 

#### Core Mechanisms:

Encoder-Only Architecture: BERT is strictly a "Reader." It does not generate text (it doesn't have a Decoder); it only processes and understands input text.

Self-Attention: This is the engine of BERT. It allows every word in a sentence to "look at" every other word to determine context.



Pre-training: BERT was trained on the entire English Wikipedia (2.5 billion words) and BookCorpus (800 million words). It already "knows" English grammar, syntax, and facts before you even start training.
#### How BERT Works in this Project
Adapted BERT to do more than just read statements.It reads Metadata as well.

A. The Input

Standard BERT models just take a sentence.The model takes a composite string.The input looks like this:

"[CLS] Barack Obama (Democrat) stated in Campaign Speech: We have cut taxes... [SEP]"

[CLS] Token: (Classification) Added at the start. This special token acts as a "bucket" that collects the mathematical meaning of the entire sequence.

[SEP] Token: (Separator) Tells BERT where the input ends.

By putting the Speaker and Party inside the text string, BERT's self-attention mechanism learns relationships like:

"If Speaker X says Y, it is usually False."

"If Topic Z is discussed in a 'Facebook Post', it has a higher probability of being Fake."

B.Binary Classification

The model used is the bert-base-uncased with a Binary Classification Head.

Tokenization:text is broken into "WordPieces"

Encoding: These pieces are converted into vectors (lists of numbers) of size 768.

Processing: The vectors pass through 12 layers of "Self-Attention," where they update each other based on context.

The classification (The Judge):

The model ignores all the word vectors except for the [CLS] vector.

This single vector (representing the whole claim + context) is passed to a simple Linear Layer.

Output: Two numbers (Logits) representing Real vs. Fake.

### RoBERTa (Robustly Optimized BERT Approach)

RoBERTa builds on BERT's architecture but optimizes the training process. It uses dynamic masking (changing the masked token during training epochs) and was trained on a much larger corpus (160GB vs 16GB). This allows it to better understand complex sentence structures and negations, which is crucial for fact-checking nuances that BERT might miss.

The Input Structure:
It does not simply feed the claim. It constructs a composite input string

*  Token: The "Classification" token at the start aggregates the semantic meaning of the entire sequence (Metadata + Statement).

* Self-Attention Layers: The model learns correlations.

* Classification Head: A simple linear layer on top of the classification token projects the embedding into 2 logits (Real vs. Fake).

### Evaluation & Performance

Benchmarked two transformer architectures on the Test Set (797 samples).

#### Comparative Metrics

<img width="710" height="135" alt="image" src="https://github.com/user-attachments/assets/ff0434e2-84b8-470b-9556-1aa1f6815d5b" />


#### Confusion Matrix Analysis

##### 1. BERT (Base Uncased)

True Negatives (Correctly Identified Fakes): 239

True Positives (Correctly Identified Reals): 323

False Positives (Fake predicted as Real): 138

False Negatives (Real predicted as Fake): 97

##### 2. RoBERTa (Base)

True Negatives (Correctly Identified Fakes): 235

True Positives (Correctly Identified Reals): 345

False Positives (Fake predicted as Real): 142

False Negatives (Real predicted as Fake): 75

##### Analysis:

RoBERTa proved superior in detecting Real News, identifying 345 legitimate claims compared to BERT's 323.

Overall, RoBERTa achieves the highest weighted F1-score (0.72), suggesting it handles the linguistic nuances of political statements better than BERT.


##### Error Analysis

A deep dive into the misclassifications on the validation dataset to understand model behavior.

###### 1. Misclassification Breakdown

BERT: Misclassified 235 samples.

RoBERTa: Misclassified 217 samples.

###### 2. Pattern Comparison

Common Errors (175 samples): Both models failed on this subset, indicating these examples are inherently difficult.

###### Distinct Errors:

BERT Only Correct: BERT correctly classified 42 samples that RoBERTa missed.

RoBERTa Only Correct: RoBERTa correctly classified 60 samples that BERT missed.

Insight: Each model demonstrates unique strengths. This lack of complete overlap suggests that an Ensemble Method (averaging predictions) could potentially boost accuracy further.

###### 3. Metadata Bias in Errors

The distribution of 'Speaker', 'Party', and 'Context' within the misclassified samples (visualized below).


Bert Miscalculation
<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/0864b83d-c834-4a25-bafc-75753a410d74" />


Roberta miscalculation
<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/0ae1e82e-287d-4a3a-b1e4-1961e360b37b" />


Contextual Challenges: Certain contexts (e.g., "Campaign Ads", "Tweets") appeared disproportionately in errors, suggesting the models struggle with the informal or hyperbolic language often used in these settings.

Speaker Bias: High-profile speakers from both parties appeared frequently in misclassifications, likely due to the complexity and volume of their statements rather than inherent model bias against a specific party.

###### 4. Qualitative Failure Modes

Specific failure types identified during manual review:

The "Sarcasm" Trap: Models struggle to detect irony. A sarcastic statement made by a truthful speaker might be flagged as Fake because the text looks like a lie.

Temporal Knowledge Gap: The model does not have access to real-time data. If a statement was true in 2012 but false in 2024, it relies on linguistic patterns rather than factual verification.

Source Overfitting: Because we inject Speaker names, the model may over-rely on the speaker's history rather than the specific statement content.
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



## Usage

streamlit run app_bert.py


### Testing the Model:

Select a pre-loaded example from the dropdown to see how the model handles known True/Fake claims.

Choose "Custom Input" to enter your own text. Try varying the Speaker field to see how the model's confidence changes based on the source!
