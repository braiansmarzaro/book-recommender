# book-recommender

- [book-recommender](#book-recommender)
  - [Objective](#objective)
  - [Dataset](#dataset)
  - [Features](#features)
  - [Steps](#steps)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [Usage](#usage)
    - [Semantic Search](#semantic-search)
    - [Browse Mode](#browse-mode)

## Objective

A book recommendation system built using Python, Langchain, ChromaDB and Pandas.

## Dataset

[Kaggle Dataset](https://www.kaggle.com/datasets/elvinrustam/books-dataset)

## Features

- [x] Data loading from environment variable
- [x] Semantic Search for book recommendations
- [x] Embedding generation using HuggingFace and LangChain
- [x] Vector storage with Chroma
- [x] Book recommendation based on user favorites
- [x] User-interface with Streamlit
- [x] Advanced filters (category, price, year, publisher, author)
- [x] Browse mode for exploring the dataset

## Steps

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Chroma Vector DB Setup
4. Embedding Generation with HuggingFace and LangChain
5. Semantic Search Implementation
6. Streamlit Frontend Development

## Installation

1. Install dependencies:
```bash
uv pip install -e .
```

## Running the Application

1. Make sure you have a `.env` file with the `DATASET_PATH` variable pointing to your books dataset CSV file.

2. Ensure the ChromaDB vector store is populated (run the notebook cells to create it if needed).

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

### Semantic Search
1. Enter a search query describing the type of book you're looking for
2. Apply optional filters in the sidebar (category, price range, publication year, publisher, author)
3. Click the "Search" button
4. View the results in a table format
5. Download results as CSV if needed

### Browse Mode
1. Check the "Show filtered dataset" option at the bottom
2. Use sidebar filters to narrow down books
3. Adjust the number of books to display