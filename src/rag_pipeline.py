# src/rag_pipeline.py
from typing import List, Dict, Any, Optional
import os
import logging
from pathlib import Path

from langchain.document_loaders import TextLoader, DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self, 
        persist_directory: str = "chroma_customer_support",
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vectorstore = None
        
    def load_additional_documents(self, data_sources: Dict[str, Any]) -> List[Document]:
        """
        Load additional documents from various sources
        
        Args:
            data_sources: Dictionary with source types and paths/URLs
        
        Returns:
            List of documents
        """
        all_documents = []
        
        # Load individual files
        if "files" in data_sources:
            for file_path in data_sources["files"]:
                try:
                    loader = TextLoader(file_path)
                    all_documents.extend(loader.load())
                    logger.info(f"Loaded document: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        # Load directories
        if "directories" in data_sources:
            for dir_path in data_sources["directories"]:
                try:
                    loader = DirectoryLoader(dir_path, glob="**/*.txt")
                    all_documents.extend(loader.load())
                    logger.info(f"Loaded directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error loading directory {dir_path}: {e}")
        
        # Load web pages (e.g., Intryc )
        if "urls" in data_sources:
            try:
                loader = WebBaseLoader(data_sources["urls"])
                all_documents.extend(loader.load())
                logger.info(f"Loaded {len(data_sources['urls'])} URLs")
            except Exception as e:
                logger.error(f"Error loading URLs: {e}")
        
        return all_documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def initialize(self, data_sources: Dict[str, Any] = None) -> None:
        """Initialize the RAG pipeline, using existing vector store if available"""
        # Check if the vector store already exists
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="customer_support"
            )
            
            # If additional data sources are provided, add them to the existing vectorstore
            if data_sources:
                additional_docs = self.load_additional_documents(data_sources)
                if additional_docs:
                    chunks = self.process_documents(additional_docs)
                    logger.info(f"Adding {len(chunks)} document chunks to existing vector store")
                    self.vectorstore.add_documents(chunks)
                    self.vectorstore.persist()
        else:
            logger.warning(f"Vector store not found at {self.persist_directory}. Please run get_data_n_clean_it.py first.")
            raise FileNotFoundError(f"Vector store not found at {self.persist_directory}")
    
    def get_retriever(self, search_type: str = "similarity", k: int = 3):
        """Get the retriever from the vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )