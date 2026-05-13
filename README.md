# Spam/Ham Email Classifier

A Machine Learning based NLP project that classifies messages as Spam or Ham using TF-IDF vectorization and Naive Bayes algorithm.

---

## Features

- Text preprocessing
- Stopword removal
- TF-IDF feature extraction
- Spam/Ham prediction
- Streamlit web interface
- Model saving using Pickle
- GitHub deployment

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit

---

## Machine Learning Concepts

### TF-IDF Vectorization

The project converts text messages into numerical vectors using TF-IDF.

Conceptually:

TF-IDF(t,d) = TF(t,d) × IDF(t)

This helps identify important words in spam messages.

Example spam keywords:
- free
- winner
- claim
- prize

---

## Algorithm Used

### Naive Bayes Classifier

Multinomial Naive Bayes is used for text classification because it performs efficiently on NLP datasets.

---

## Dataset

Dataset used:
SMS Spam Collection Dataset

Contains:
- Spam messages
- Ham (normal) messages

---

## Project Structure

```bash
spam-email-classifier/
│
├── spam.csv
├── train_model.py
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Train the model

```bash
python train_model.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## Example Predictions

### Spam Message

```text
Congratulations! You won a free iPhone.
```

Prediction:
```text
SPAM
```

---

### Ham Message

```text
Hey bro are we meeting tomorrow?
```

Prediction:
```text
HAM
```

---

## Model Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Future Improvements

- Lemmatization
- Stemming
- Logistic Regression comparison
- SVM classifier
- Deep Learning models
- Email API integration

---

## Author

ASHWIN SIVAGNANAM R
