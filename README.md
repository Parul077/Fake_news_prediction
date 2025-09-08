# 📰 Fake News Prediction

A Machine Learning project that predicts whether a given news article is real or fake.
The model uses Natural Language Processing (NLP) techniques for text preprocessing and classification.

## 🚀 Features

📑 Text Preprocessing: Tokenization, stopword removal, stemming.

📊 TF-IDF Vectorization: Converts text into numerical feature vectors.

🤖 Machine Learning Models: Trained classifiers like Logistic Regression, Naive Bayes, and PassiveAggressiveClassifier.

✅ Accuracy Evaluation: Compares multiple models for best performance.

## 🛠️ Tech Stack

Language: Python

Libraries:

Pandas, NumPy → Data Handling

NLTK, re → Text Preprocessing

Scikit-learn → ML Models & TF-IDF Vectorizer

Jupyter Notebook → Development

## 📂 Project Structure
├── Fake_News_Prediction.ipynb   # Main Jupyter notebook
├── News.csv                    #  dataset 
├── README.md                    # Project documentation

## 📊 Dataset

Source: Kaggle Fake News Dataset

Columns:

id → Unique ID

title → Headline of the news

author → Author of the article

text → Full article text

label → Target variable (1 = Fake, 0 = Real)

## ⚙️ Installation & Setup

Clone the repository

git clone https://github.com/Parul077/Fake_news_prediction.git
cd fake-news-prediction


Install required libraries

pip install pandas numpy scikit-learn nltk


Open the Jupyter Notebook

jupyter notebook Fake_News_Prediction.ipynb


Run all cells to train and test the model.

## 📈 Model Training

Text Cleaning – Removing punctuation, stopwords, and applying stemming.

Feature Extraction – Using TF-IDF Vectorization.

Model Training – Logistic Regression / Passive Aggressive Classifier.

Evaluation – Accuracy, Precision, Recall, F1-score.

## 🔮 Results

Logistic Regression: ~92% accuracy

PassiveAggressiveClassifier: ~93% accuracy

(Your results may vary depending on dataset split and preprocessing.)


👩‍💻 Author

Developed by Parul Singh ✨
