# main.py
import os
import argparse
from src.llm_providers import AnthropicLLM, OpenAILLM
from src.agent import ResearchAssistant
from src.tools.calculator import Calculator
from src.tools.web_search import WebSearch
from src.rag_pipeline import RAGPipeline
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def setup_vector_store(documents_dir, force_rebuild=False):
    """Set up or load the vector store"""
    persist_directory = "chroma_db"
    
    # Check if the vector store already exists and we're not forcing a rebuild
    if os.path.exists(persist_directory) and os.listdir(persist_directory) and not force_rebuild:
        print(f"Loading existing vector store from {persist_directory}")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        return vectorstore
    
    # Build new vector store
    print(f"Building new vector store in {persist_directory}")
    
    # Load documents
    loader = DirectoryLoader(documents_dir, glob="**/*.pdf")
    documents = loader.load()
    
    if not documents:
        print(f"No documents found in {documents_dir}. Creating dummy document for testing.")
        from langchain.schema import Document
        documents = [Document(page_content="This is a test document for the research assistant.", metadata={"source": "test"})]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, # or anyother 
        chunk_overlap=120 # 
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Research Assistant')
    parser.add_argument('--llm', choices=['openai', 'anthropic'], default='openai', help='LLM provider')
    parser.add_argument('--documents', default='data/documents/', help='Path to document directory')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of vector store')
    args = parser.parse_args()
    
    # Initialize LLM provider based on user choice
    if args.llm == 'openai':
        # Ensure API key is set
        if 'OPENAI_API_KEY' not in os.environ:
            api_key = input("Enter your OpenAI API key: ")
            os.environ['OPENAI_API_KEY'] = api_key
        llm = OpenAILLM(model_name="gpt-3.5-turbo")
    else:
        # Ensure API key is set
        if 'ANTHROPIC_API_KEY' not in os.environ:
            api_key = input("Enter your Anthropic API key: ")
            os.environ['ANTHROPIC_API_KEY'] = api_key
        llm = AnthropicLLM(model_name="claude-3-sonnet-20240229")
    
    # Set up vector store and retriever
    try:
        vectorstore = setup_vector_store(args.documents, args.rebuild)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        print("Using a RAG Pipeline instead...")
        
        try:
            # Try to initialize the RAG pipeline
            rag = RAGPipeline()
            rag.initialize()
            retriever = rag.get_retriever()
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            # Create a dummy retriever
            from langchain.schema import Document
            class DummyRetriever:
                def get_relevant_documents(self, query):
                    return [Document(page_content="This is a dummy document for testing purposes.", metadata={"source": "dummy"})]
            retriever = DummyRetriever()
    
    # Initialize tools
    tools = {
        "calculator": Calculator(),
        "web_search": WebSearch()
    }
    
    # Initialize agent
    agent = ResearchAssistant(
        llm_provider=llm,
        retriever=retriever,
        tools=tools
    )
    
    # Interactive loop
    print("\n======================================")
    print("📚 LLM-Powered Research Assistant 🤖")
    print("======================================")
    print("Ask questions or type 'exit' to quit.")
    print("Type 'help' for command options.\n")
    
    while True:
        query = input("\n📝 Your question: ")
        if query.lower() in ('exit', 'quit'):
            break
        elif query.lower() == 'help':
            print("\nCommands:")
            print("  'exit' or 'quit' - Exit the application")
            print("  'help' - Show this help message")
            print("\nExample questions:")
            print("  - What is machine learning?")
            print("  - Calculate 15 * 7.5")
            print("  - Search for recent breakthroughs in quantum computing")
            continue
            
        try:
            result = agent.process_query(query)
            print("\n🤖 Answer:", result["response"])
            print("\nQuery type:", result["query_type"])
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")

if __name__ == "__main__":
    main()