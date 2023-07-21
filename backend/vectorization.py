import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils import save_data, load_data

def vectorize_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    store_name = "vector_store"
    if os.path.exists(f"{store_name}.pkl"):
        vector_store = load_data(f"{store_name}.pkl")
    else:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        save_data(vector_store, f"{store_name}.pkl")
    return chunks, vector_store
