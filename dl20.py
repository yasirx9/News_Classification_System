import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('trainingDataset.csv')
X = df['News']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

model = Sequential()
model.add(Dense(128, input_dim=X_train_tfidf.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

history = model.fit(X_train_tfidf, y_train_encoded, epochs=20, batch_size=32, validation_data=(X_test_tfidf, y_test_encoded))

y_pred = model.predict(X_test_tfidf)
y_pred_class = encoder.inverse_transform(y_pred.argmax(axis=1))

accuracy = (y_pred_class == y_test).mean()
print(f"\n[Deep Learning Model's Accuracy for 10 Epochs: {accuracy * 100:.2f}%]\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_title('Epoch Accuracy Diagram for 20 Epochs')
ax1.legend()

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.set_title('Epoch Loss Diagram for 20 Epochs')
ax2.legend()

plt.show()

while True:
  new_news = input("Enter a news: ")
  new_news_tfidf = tfidf.transform([new_news]).toarray()
  predicted_category = encoder.inverse_transform(model.predict(new_news_tfidf).argmax(axis=1))
  print(f"Predicted Category: {predicted_category[0]}\n")
