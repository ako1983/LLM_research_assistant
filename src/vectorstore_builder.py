#!/usr/bin/env python
# coding: utf-8

# from datasets import load_dataset

# ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

import os
from openai import OpenAI

# === Load OpenAI API Key from File ===
def load_openai_key(filepath="openai_api_key.txt"):
    with open(filepath, "r", encoding="utf-8") as file:
        key = file.read().strip()
        os.environ["OPENAI_API_KEY"] = key
        return key

# Load API Key into environment and create OpenAI client
openai_api_key = load_openai_key()
client = OpenAI(api_key=openai_api_key)

# --- STEP 0: Ensure Required Packages are Installed ---
import importlib.util
import sys
import subprocess
import os
import pandas as pd
import re
import tiktoken

# def install_if_missing(package_name, import_name=None):
#     """Install package if not already installed."""
#     import_name = import_name or package_name
#     if importlib.util.find_spec(import_name) is None:
#         print(f"Package {package_name} not found. Installing...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
#     else:
#         print(f"Package {package_name} is already installed.")

# required_packages = {
#     "pandas": "pandas",
#     "langchain": "langchain",
#     "openai": "openai",
#     "chromadb": "chromadb"  # Underlying package used by Chroma vector store.
# }

# for pkg, imp_name in required_packages.items():
#     install_if_missing(pkg, imp_name)


# --- STEP 1: Data Preparation (Cleaning) ---

raw_file = /Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
# i have the data here downloaded and saved as sample data
#raw_file = "sample_data.csv"
cleaned_file = "cleaned_data.csv"

# Check if cleaned file already exists
if not os.path.exists(cleaned_file):
    print("\nCleaned file not found. Loading raw data from:", raw_file)
    df = pd.read_csv(raw_file)
    print("Initial dataset shape:", df.shape)
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    print("Shape after dropping duplicates:", df.shape)
    
    # Remove rows with missing values
    df = df.dropna().reset_index(drop=True)
    print("Shape after dropping rows with missing values:", df.shape)
    
    # Identify text columns and define a helper function to clean text
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    def clean_text(text):
        # Remove non-ASCII characters, replace multiple spaces/newlines with a single space, and strip extra whitespace
        text = text.encode('ascii', errors='ignore').decode()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    for col in text_columns:
        df[col] = df[col].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
        # Optional: convert text to lowercase for uniformity
        df[col] = df[col].str.lower()
        df.columns = df.columns.str.lower()
        
    # Save the cleaned dataset
    df.to_csv(cleaned_file, index=False)
    print("Cleaned dataset saved to:", cleaned_file)
else:
    print("\nCleaned file exists. Loading cleaned data.")
    df = pd.read_csv(cleaned_file)

# --- STEP 2: Generate Embeddings & Index Data in Chroma Vector Store ---
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma


persist_directory = "chroma_customer_support"

# Check if the Chroma vector store already exists (folder exists and is non-empty)
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    print("\nVector store not found. Creating a new vector store.")
    
    # Prepare documents: combine 'instruction' and 'response'
    documents = []
    metadatas = []
    for _, row in df.iterrows():
        doc_text = f"Instruction: {row['instruction']}\nResponse: {row['response']}"
        documents.append(doc_text)
        metadata = {
            "flags": row["flags"],
            "category": row["category"],
            "intent": row["intent"]
        }
        metadatas.append(metadata)
    print("Total documents prepared:", len(documents))
   
    # Create and persist the Chroma vector store
    vector_store = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="customer_support",
        persist_directory=persist_directory
    )
    vector_store.persist()
    print("Vector store has been created and persisted at:", persist_directory)
else:
    print("\nVector store exists. Loading vector store.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) # OpenAI embeddings-or any other you prefer
    # optionally add  model="text-embedding-3-small"
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="customer_support"
    )

# --- STEP 3: Test Retrieval with a Sample Query ---
query = "how do i fix a wi-fi connection issue?"
results = vector_store.similarity_search(query, k=3)
print("\nTop 3 retrieved documents for the query:")
for result in results:
    print("Document:\n", result.page_content)
    print("Metadata:", result.metadata)
    print("-" * 40)

"""
import dspy

class RAGPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_optimizer = QueryOptimizer()
        self.answer_generator = AnswerGenerator()
        self.response_evaluator = ResponseEvaluator()

    def forward(self, query, retrieved_context):
        optimized_query = self.query_optimizer.forward(query)
        response = self.answer_generator.forward(context=retrieved_context, question=optimized_query.reformulated_query)
        evaluation = self.response_evaluator.forward(answer=response.answer, question=query)
        return response.answer, evaluation.score

class ReformulateQuery(dspy.Signature):
    """Reformulates a user query to improve retrieval accuracy."""
    original_query = dspy.InputField()
    reformulated_query = dspy.OutputField()

class QueryOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rewriter = dspy.Predict(ReformulateQuery)  # A DSPy component for rewriting queries

    def forward(self, query):
        return self.rewriter(original_query=query)
"""
