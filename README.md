# 🖼️ Smart Screenshot Search Engine

Smart Screenshot Search Engine is a project I developed to make it easier to find and organize screenshots from a large collection.

Instead of manually checking screenshots one by one, the system allows users to search screenshots using the text present in them, their meaning, and their visual content.

The project uses **OCR, semantic search, visual search using CLIP, and CNN-based classification**.

## Features

* **OCR Search** – Extracts text from screenshots using EasyOCR and allows searching based on that text.
* **Semantic Search** – Finds screenshots based on the meaning of the search query using Sentence Transformers.
* **Visual Search** – Finds visually similar screenshots using CLIP.
* **CNN Classification** – Classifies screenshots into six predefined categories.
* **Category Filtering** – Allows screenshots to be explored category-wise.
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

### 3. Visual Search

For visual search, **CLIP** is used to generate image embeddings.

The visual embedding of the query/image is compared with screenshot embeddings to find visually similar screenshots.

### 4. CNN Classification

A CNN model built using **PyTorch** is used to classify screenshots into the six categories.

The dataset was divided into training, validation and testing sets, and data augmentation was used during training.

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
```

## Running the Project

First install the required Python libraries:

```bash
pip install -r requirements.txt
```

Then run the Streamlit application:

```bash
streamlit run UI/app.py
```

The application will open in the browser.

## Project Interface

The Streamlit interface provides:

* Search settings
* Category filtering
* Number of results selection
* Dataset overview
* Category distribution
* Screenshot search results
* OCR text viewing
* Full screenshot viewing

## Project Goal

The main goal of this project is to make screenshot collections easier to search.

A user should be able to enter a query such as a **place, product, recipe, receipt, ticket, or other information**, and get the most relevant screenshots without manually going through the entire collection.

## Future Improvements

Some possible improvements for the project are:

* Improve CNN classification accuracy
* Add support for more screenshot categories
* Improve OCR accuracy for different fonts and languages
* Improve the visual and semantic search combination
* Deploy the application for easier access

## Demo

The project is also deployed using Streamlit.

**Live Demo:**
https://smart-screenshot-search-engine-hdfwwsszixlnz3uo2bff2z.streamlit.app/

## Author

**Priyanka Ghogare**

Computer Engineering Student

This project was developed as part of my academic/project work.
