import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('trainingDataset.csv')

X = df['News']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n[Random Forest Model's Accuracy: {accuracy * 100:.2f}%]\n")

while True:
  new_news = input("Enter a news: ")
  new_news_tfidf = tfidf.transform([new_news])
  predicted_category = model.predict(new_news_tfidf)
  print(f"Predicted Category: {predicted_category[0]}\n")
