import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('trainingDataset.csv')

tfidf = TfidfVectorizer(stop_words='english')

X = tfidf.fit_transform(data['News'])
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'[Logistic Regression Model\'s Accuracy: {accuracy * 100:.2f}%]')

while True:
  new_news = input("Enter a news: ")
  new_news_tfidf = tfidf.transform([new_news])
  predicted_category = model.predict(new_news_tfidf)
  print(f"Predicted Category: {predicted_category[0]}\n")
