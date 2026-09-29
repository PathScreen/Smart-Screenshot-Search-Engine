# 🔍 VisionX

**AI-powered screenshot search system that retrieves screenshots using their text, meaning, and visual content.**

## 🚀 Live Demo

**Try VisionX:**  
https://smart-screenshot-search-engine-hdfwwsszixlnz3uo2bff2z.streamlit.app/

## 🖼️ Application Preview

<!-- Add your UI screenshot here after confirming its repository path -->

## 📑 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [How the System Works](#how-the-system-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [CNN Classification Experiment](#cnn-classification-experiment)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [References](#references)

## ✨ Features

- **OCR Search** – Extracts text from screenshots using EasyOCR.
- **Semantic Search** – Finds screenshots based on the meaning of a text query using Sentence Transformers.
- **Visual Search** – Finds visually relevant screenshots using CLIP image embeddings.
- **Category Filtering** – Allows screenshots to be explored category-wise.
- **Upload New Screenshots** – Users can upload a screenshot and select its category.
- **Automatic Indexing** – Uploaded screenshots are processed and added to the search indexes.
- **Streamlit Interface** – Provides a simple interface for searching and viewing screenshots.

## 📊 Dataset

VisionX uses a collection of **4000 screenshots** organized into six categories:

1. Addresses & Locations
2. Interior Design
3. Products & Shopping
4. Receipts & Bills
5. Recipes
6. Tickets & Bookings

The screenshots contain different types of information such as places, products, receipts, recipes, and booking details.

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- PyTorch
- EasyOCR
- Sentence Transformers
- CLIP
- Scikit-learn
- NumPy
- Pandas
- Streamlit

## ⚙️ How the System Works

### 1. OCR

EasyOCR extracts the text present in the screenshots.

The extracted OCR text is stored as metadata and is used for text-based searching.

### 2. Semantic Search

The OCR text is converted into numerical embeddings using the **Sentence Transformers `all-MiniLM-L6-v2`** model.

When a user enters a text query, the query is also converted into an embedding. Cosine similarity is then used to find screenshots with similar meaning.

Exact matches in the extracted OCR text are also considered while ranking the results.

### 3. Visual Search

For visual search, **CLIP** is used to generate image embeddings.

These embeddings are compared with the stored screenshot embeddings to find visually relevant screenshots.

VisionX uses the **`openai/clip-vit-base-patch32`** model.

### 4. Upload and Indexing

Users can upload a new screenshot through the application and select its category.

After uploading, the system:

1. Saves the screenshot in the selected category.
2. Extracts text using EasyOCR.
3. Generates a semantic embedding using Sentence Transformers.
4. Generates a visual embedding using CLIP.
5. Updates the metadata and embedding indexes.

The uploaded screenshot can then be searched through the application.

## 🚀 Getting Started

### Prerequisites

Make sure the following are installed:

- Python 3.10 or later
- Git
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/PathScreen/Smart-Screenshot-Search-Engine.git
