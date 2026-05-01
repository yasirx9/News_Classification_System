# News Classification System

> A multi-algorithm **News Article Classification System** that categorizes news articles into topics using both traditional Machine Learning and Deep Learning approaches. Includes a web scraper for collecting real news data.

---

## 📌 Overview

This system automatically classifies news articles into predefined categories (e.g., Sports, Politics, Technology, Entertainment). It compares the performance of multiple classification techniques — from classical ML models to neural networks trained over different epoch counts — and visualizes the results.

---

## ✨ Features

- **Multi-Algorithm Comparison** — Evaluates KNN, Naive Bayes, Logistic Regression, Random Forest, and Deep Learning models
- **Deep Learning Models** — Neural network trained at 10 and 20 epochs with performance graphs
- **Web Scraper** — Automated scraper to collect fresh news articles for the dataset
- **Performance Visualization** — Accuracy/loss graphs plotted and saved as images
- **Comprehensive Report** — Detailed project report included (`Project_Report.docx`)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | Classical ML classifiers |
| TensorFlow / Keras | Deep learning models |
| Pandas / NumPy | Data processing |
| Matplotlib | Performance visualization |
| BeautifulSoup / Requests | Web scraping |

---

## 📁 Repository Structure

```
News_Classification_System/
├── scraper/                    # Web scraper to collect news articles
├── knn.py                      # K-Nearest Neighbors classifier
├── nb.py                       # Naive Bayes classifier
├── logReg.py                   # Logistic Regression classifier
├── rf.py                       # Random Forest classifier
├── dl10.py                     # Deep Learning model (10 epochs)
├── dl20.py                     # Deep Learning model (20 epochs)
├── graph.py                    # Accuracy/loss graph plotting
├── trainingDataset.csv         # Labeled training dataset
├── epochs10.png                # Performance graph (10 epochs)
├── epochs20.png                # Performance graph (20 epochs)
└── Project_Report.docx         # Full project report
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install scikit-learn tensorflow pandas numpy matplotlib requests beautifulsoup4
```

### Run a Classifier

```bash
# Run Naive Bayes
python nb.py

# Run Logistic Regression
python logReg.py

# Run K-Nearest Neighbors
python knn.py

# Run Random Forest
python rf.py
```

### Train Deep Learning Models

```bash
# Train with 10 epochs
python dl10.py

# Train with 20 epochs
python dl20.py
```

### Visualize Results

```bash
python graph.py
```

### Collect Fresh News Data

```bash
cd scraper/
python scraper.py   # (run the scraper script inside the folder)
```

---

## 📊 Results

| Model | Notes |
|---|---|
| Naive Bayes | Fast, good baseline for text classification |
| Logistic Regression | Strong linear classifier for NLP tasks |
| KNN | Distance-based; slower on large datasets |
| Random Forest | Ensemble method with robust performance |
| Deep Learning (10 epochs) | See `epochs10.png` for training curves |
| Deep Learning (20 epochs) | See `epochs20.png` for training curves |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

Made by [Muhammad Yasir](https://yasirportfolio.page.gd)
