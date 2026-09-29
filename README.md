# 🔍 VisionX

**AI-powered screenshot search system that retrieves screenshots using their text, meaning, and visual content.**

## 🚀 Live Demo

**Try VisionX:**  
https://smart-screenshot-search-engine-hdfwwsszixlnz3uo2bff2z.streamlit.app/

## 🖼️ Application Preview

<img width="1917" height="970" alt="Screenshot 2026-09-30 003913" src="https://github.com/user-attachments/assets/25169878-76a6-4f0f-945f-893ad01d5b75" />

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
```

2. Navigate to the project directory:

```bash
cd Smart-Screenshot-Search-Engine
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

Start the Streamlit application:

```bash
streamlit run UI/app.py
```

The application will open in the browser.

### Searching

1. Enter a search query.
2. VisionX processes the query.
3. Relevant screenshots are retrieved using semantic or visual similarity.
4. Use category filtering to explore screenshots by category.

### Uploading a Screenshot

1. Select the upload option.
2. Choose a screenshot.
3. Select its category.
4. Upload the screenshot.
5. VisionX processes the screenshot using OCR, MiniLM, and CLIP.
6. The screenshot is added to the search indexes.

## 📁 Project Structure

```text
Smart-Screenshot-Search-Engine/
│
├── Data/
│   ├── Addresses_Locations/
│   ├── Interior_Design/
│   ├── Products_Shopping/
│   ├── Receipts_Bills/
│   ├── Recipes/
│   └── Tickets_Bookings/
│
├── UI/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🧠 CNN Classification Experiment

A separate CNN experiment was conducted to study screenshot classification across the six dataset categories.

The CNN experiment is separate from the main semantic and visual retrieval pipeline.

## ⚠️ Limitations

- OCR performance depends on screenshot quality.
- Large datasets require more computational resources.
- Search results depend on the available dataset.
- Semantic search quality depends on query clarity.
- Visual search depends on the visual similarity between the query and screenshots.

## 🔮 Future Scope

- Automatic screenshot category classification using CNN.
- Mobile and camera-based screenshot search.
- Voice and multilingual search.
- Cloud-based storage and indexing.
- Improved retrieval models.
- Continuous learning from newly added screenshots.

## 📚 References

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/)
