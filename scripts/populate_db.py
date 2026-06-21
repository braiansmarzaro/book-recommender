from dotenv import load_dotenv
import pandas as pd
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
import chromadb

load_dotenv()
import os

from langchain_core.documents import Document
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

vector_store = Chroma(
    persist_directory="./chroma_langchain_db",
    collection_name="books",
    embedding_function=embeddings,
)
dataset_path = os.getenv("DATASET_PATH")
df = pd.read_csv(dataset_path)
df.loc[df.Authors.str.startswith("By ", na=False), "Authors"] = df.loc[
    df.Authors.str.startswith("By ", na=False), "Authors"
].str[3:]


df_with_desc = df[df.Description.notna()].copy()

# Calculate 20% of the dataframe
sample_size = int(len(df_with_desc) * 1)

# Sort by multiple columns to get diverse sample
# Sort by category, year, and price to maximize diversity
df_sorted = df_with_desc.sort_values(
    by=["Category", "Publish Date (Year)", "Price Starting With ($)"],
    na_position="first",
)

# Take evenly spaced samples to maximize diversity
step = len(df_sorted) // sample_size
df_sample = df_sorted.iloc[::step][:sample_size]

print(f"Sampling {len(df_sample)} books out of {len(df_with_desc)} (20%)")

# Add sampled documents to vector store
for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    doc = Document(
        page_content=f"Title: {row['Title']} - Description: {row['Description']} - Category: {row['Category'] if pd.notna(row['Category']) else 'N/A'}",
        metadata={
            "title": row["Title"],
            "authors": row["Authors"],
            "category": row["Category"] if pd.notna(row["Category"]) else "",
            "publisher": row["Publisher"] if pd.notna(row["Publisher"]) else "",
            "price": row["Price Starting With ($)"],
            "publish_year": row["Publish Date (Year)"],
        },
    )
    vector_store.add_documents([doc])

print(f"Successfully added {len(df_sample)} diverse books to the vector store")
