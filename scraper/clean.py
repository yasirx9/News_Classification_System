import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
  words = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  lemmatizer = WordNetLemmatizer()
  cleaned_words = [ lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
  cleaned_text = ' '.join(cleaned_words)
  return cleaned_text

try:
  df = pd.read_csv('dataset.csv')
  df['News'] = df['News'].apply(clean_text)
  df.dropna
  df.to_csv('trainingDataset.csv', index=False)
  print("Dataset cleaned, stopwords removed, and lemmatized. Saved to 'trainingDataset.csv'")
except Exception as e:
  print(e)
