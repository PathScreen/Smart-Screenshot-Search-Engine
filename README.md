# 🔍 VisionX

VisionX is a project I developed to make it easier to find and organize screenshots from a large collection.

Instead of manually checking screenshots one by one, the system allows users to search screenshots using the text present in them, their meaning, and their visual content.

The project uses **OCR, semantic search, visual search using CLIP, and CNN-based classification**.

## 🚀 Live Demo

**Try VisionX:**  
https://smart-screenshot-search-engine-hdfwwsszixlnz3uo2bff2z.streamlit.app/

## Features

* **OCR Search** – Extracts text from screenshots using EasyOCR and allows searching based on that text.
* **Semantic Search** – Finds screenshots based on the meaning of the search query using Sentence Transformers.
* **Visual Search** – Uses CLIP to find visually relevant screenshots.
* **CNN Classification** – Classifies screenshots into six predefined categories.
* **Category Filtering** – Allows screenshots to be explored category-wise.
* **Upload New Screenshots** – Users can upload a new screenshot and select its category.
* **Automatic Indexing** – Uploaded screenshots are processed using OCR and embedding models and added to the search indexes.
* **Streamlit Interface** – Provides the user interface for searching and viewing screenshots.

## Dataset

The project uses a collection of **4000 screenshots** organized into six categories:

1. Addresses & Locations
2. Interior Design
3. Products & Shopping
4. Receipts & Bills
5. Recipes
6. Tickets & Bookings

The screenshots are mainly phone screenshots containing different types of information such as places, products, receipts, recipes, and booking details.

## Technologies Used

* Python
* Jupyter Notebook
* PyTorch
* EasyOCR
* Sentence Transformers
* CLIP
* Scikit-learn
* NumPy
* Pandas
* Streamlit

## How the System Works

### 1. OCR

EasyOCR is used to extract the text present in the screenshots.

The extracted text is stored as metadata and is used for text-based searching.

### 2. Semantic Search

The extracted OCR text is converted into numerical embeddings using the **Sentence Transformers `all-MiniLM-L6-v2` model**.

When a user enters a search query, the query is also converted into an embedding. Cosine similarity is then used to find screenshots with similar meaning.

Exact matches in the extracted OCR text are also given importance while ranking the results.

### 3. Visual Search

For visual search, **CLIP** is used to generate image embeddings.

These embeddings are compared with the stored screenshot embeddings to find visually relevant screenshots.

The project uses the **`openai/clip-vit-base-patch32`** model.

### 4. CNN Classification

A CNN model built using **PyTorch** is used to classify screenshots into the six categories.

The dataset was divided into training, validation and testing sets, and data augmentation was used during training.

### 5. Upload and Indexing

Users can upload a new screenshot through the application and select its category.

After uploading, the system:

1. Saves the screenshot in the selected category.
2. Extracts text using EasyOCR.
3. Generates a semantic embedding using Sentence Transformers.
4. Generates a visual embedding using CLIP.
5. Updates the metadata and embedding indexes.

The uploaded screenshot can then be searched in the current application.

## Project Structure

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
