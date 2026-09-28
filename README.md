<div align="center">

# 🛒 E-Commerce Product Review Sentiment Analyzer

**Classify customer reviews as Positive, Negative, or Neutral with NLP, TF-IDF, and a Linear SVM, served through a dark-themed Gradio web app.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK-154F5B)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Model Evaluation](#-model-evaluation)
- [Example Predictions](#-example-predictions)
- [Known Limitations](#-known-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 🔎 Overview

Online stores receive thousands of reviews, and reading them all is not practical. This project trains a sentiment classifier on a 100k-row e-commerce review dataset and wraps it in an interactive web app. Enter a product category, product name, main aspect (such as *Battery* or *Sound*), and a review, and the app returns the predicted sentiment with a confidence score.

## ✨ Features

- **Text preprocessing pipeline**: lowercasing, HTML and URL removal, stop-word filtering, and lemmatization
- **Sentiment-aware stop words**: negations and opinion words (*not*, *very*, *never*, *bad*, *great*, ...) are kept so meaning isn't lost
- **TF-IDF features** with unigrams and bigrams (up to 30,000 features)
- **Linear SVM** (`LinearSVC`) with balanced class weights
- **Augmented training data**: 50 hand-written examples covering negation, neutral, and mixed reviews
- **Phrase-based overrides** for strongly worded reviews
- **Full evaluation**: accuracy, precision, recall, F1, classification report, confusion matrix, and metrics chart
- **Model persistence** with `joblib`
- **Gradio UI** with a confidence bar, result card, and 50 clickable test examples

## 🖼 Demo

> Add your screenshots to an `assets/` folder and update the paths below.

| App interface | Confusion matrix |
| :---: | :---: |
| ![App screenshot](assets/app-screenshot.png) | ![Confusion matrix](assets/confusion_matrix.png) |

## 🧰 Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data handling | pandas, NumPy |
| NLP | NLTK (stop words, WordNet lemmatizer), regex |
| Machine learning | scikit-learn (`TfidfVectorizer`, `LinearSVC`) |
| Visualization | Matplotlib, Seaborn |
| Web interface | Gradio |
| Persistence | joblib |

## 📁 Project Structure

```
.
├── app.py              # Data loading, training, evaluation, and Gradio app
├── requirements.txt    # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
└── assets/             # Screenshots used in this README
```

Files generated when you run the script:

```
confusion_matrix.png
model_performance.png
ecommerce_svm_sentiment_model.pkl
ecommerce_tfidf_vectorizer.pkl
```

## 📊 Dataset

The script reads `ecommerce_product_reviews_100k.csv` from the project root. The file is not included in this repository because of its size, so place your copy next to `app.py`.

Required columns:

| Column | Description | Example |
| --- | --- | --- |
| `product_category` | Product category | `Electronics` |
| `product_name` | Product name | `Wireless Headphones` |
| `main_aspect` | Aspect the review focuses on | `Battery` |
| `review_text` | The customer's review | `The battery life is excellent.` |
| `sentiment` | Label: `positive`, `negative`, `neutral` (or `pos`, `neg`, `neu`) | `positive` |

Rows with missing values, duplicates, or unrecognized labels are dropped automatically.

## ⚙️ Installation

**Prerequisites:** Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ecommerce-review-sentiment-analyzer.git
cd ecommerce-review-sentiment-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

1. Place `ecommerce_product_reviews_100k.csv` in the project root.
2. Run the script:

```bash
python app.py
```

The script will:

1. Download the required NLTK data (first run only)
2. Load and clean the dataset
3. Train the TF-IDF + SVM model
4. Print the evaluation metrics and save the plots
5. Save the trained model and vectorizer
6. Launch the Gradio app at **http://127.0.0.1:7860**

In the app, fill in the four fields, click **⚡ ANALYZE REVIEW**, or pick any of the 50 built-in examples.

## 🧠 How It Works

```
Raw review ──► Clean text ──► TF-IDF (1-2 grams) ──► Linear SVM ──► Prediction
                                                          │
                                        Softmax on decision scores ──► Confidence
                                                          │
                                       Strong-phrase check (optional override)
```

1. **Load and clean**: drop nulls and duplicates, then normalize labels to `Positive`, `Negative`, or `Neutral`.
2. **Combine fields**: category, product name, aspect, and review text are joined into one string.
3. **Preprocess**: lowercase, strip HTML and URLs, remove non-letters, remove stop words (keeping sentiment-bearing words), and lemmatize.
4. **Augment**: append 50 hand-written examples that teach the model about negation and mixed opinions.
5. **Split**: 80% train and 20% test, stratified by sentiment, `random_state=42`.
6. **Vectorize**: `TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)`.
7. **Train**: `LinearSVC(C=1.5, class_weight="balanced", max_iter=5000)`.
8. **Predict**: the SVM's decision scores go through a softmax to produce a relative confidence value. Reviews containing very strong phrases (for example *"terrible"* or *"excellent"*) can override the prediction with a minimum confidence of 90%.

## 📈 Model Evaluation

Running `app.py` prints accuracy, weighted precision, recall, and F1, plus a per-class classification report, and saves two figures:

- `confusion_matrix.png`: counts of actual vs. predicted sentiment
- `model_performance.png`: bar chart of the four overall metrics

Add your results here after running the script:

| Metric | Score |
| --- | --- |
| Accuracy | _fill in_ |
| Precision (weighted) | _fill in_ |
| Recall (weighted) | _fill in_ |
| F1 (weighted) | _fill in_ |

## 💬 Example Predictions

| Review | Expected |
| --- | --- |
| "The battery life is excellent and amazing." | 😊 Positive |
| "The sound quality is terrible." | 😞 Negative |
| "The camera is average." | 😐 Neutral |
| "Very good sound but the battery is average." | 😊 Positive |
| "Not good. The phone is very slow." | 😞 Negative |

## ⚠️ Known Limitations

- **Confidence is not a probability.** It is a softmax over SVM decision scores, so treat it as a relative score. The phrase-override rule also raises it to at least 90%.
- **Phrase overrides are simple substring matches.** A review like "not great" contains "great" and could be pushed to Positive, and mixed reviews may be forced to one side.
- **Mixed and sarcastic reviews are hard** for a bag-of-words model.
- **English only.**
- **The dataset is not bundled**, so results depend on the data you supply.

## 🚀 Future Improvements

- Replace substring overrides with negation-aware handling or remove them
- Calibrate probabilities (for example with `CalibratedClassifierCV`)
- Add cross-validation and hyperparameter search
- Try transformer models such as DistilBERT for comparison
- Add aspect-level sentiment for multi-aspect reviews
- Deploy to Hugging Face Spaces or Docker

## 👤 Author

**Muhammed Dhilshad**
Data Science / B1 · Trycod

## 📄 License

This project is released under the [MIT License](LICENSE).
