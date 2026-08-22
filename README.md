# 🖼️ Smart Screenshot Search Engine

A smart screenshot search engine that helps users quickly find and retrieve screenshots using OCR, semantic search, visual search, and CNN-based image classification.

## 🚀 Features

- 🔍 **OCR Search** – Extracts text from screenshots using EasyOCR.
- 🧠 **Semantic Search** – Finds screenshots based on meaning using Sentence Transformers.
- 🖼️ **Visual Search** – Uses CLIP to find visually similar screenshots.
- 🤖 **CNN Classification** – Automatically classifies screenshots into predefined categories.
- 📂 **Category Filtering** – Organizes screenshots into meaningful categories.
- 🖥️ **Streamlit UI** – Provides an interactive interface for searching and viewing screenshots.

## 📁 Screenshot Categories

The project contains screenshots organized into six categories:

- Addresses & Locations
- Interior Design
- Products & Shopping
- Receipts & Bills
- Recipes
- Tickets & Bookings

## 🛠️ Technologies Used

- Python
- PyTorch
- EasyOCR
- Sentence Transformers
- CLIP
- Scikit-learn
- NumPy
- Pandas
- Streamlit
- Jupyter Notebook

## 🧠 Machine Learning Components

### OCR

EasyOCR is used to extract text from screenshots, allowing users to search screenshots based on their visible text.

### Semantic Search

Sentence Transformers are used to convert OCR text into embeddings and perform meaning-based search.

### Visual Search

CLIP embeddings are used to compare the visual content of screenshots and retrieve visually similar images.

### CNN Classification

A PyTorch-based Convolutional Neural Network is used to classify screenshots into six categories.

## 📂 Project Structure

```text
Smart-Screenshot-Search-Engine/
│
├── Data/
├── Embeddings/
├── Models/
├── NoteBooks/
├── Results/
├── UI/
├── requirements.txt
└── .gitignore
```
## ▶️ Running the Application

### 1. Install the required dependencies

```bash
pip install -r requirements.txt

streamlit run UI/app.py
