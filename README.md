# 📚 LLM-Powered Research Assistant 🤖

An advanced AI-powered research assistant that leverages Retrieval-Augmented Generation (RAG) to provide accurate responses to user queries by retrieving relevant documents and reasoning through complex questions.

## Features

- **Smart Query Routing**: Autonomously decides whether to answer directly from knowledge, retrieve additional context, or use specialized tools
- **RAG Pipeline**: Retrieves relevant documents to enhance responses with accurate, up-to-date information
- **Multi-step Reasoning**: Uses DSPy for structured reasoning to break down complex queries
- **Tool Integration**: Utilizes calculators, web search, and other external tools when needed
- **Hybrid Search**: Combines dense and sparse retrievers for optimal document retrieval
- **Multi-Modal Support**: Processes and responds to both text and image inputs
- **Error Handling**: Robust error management with graceful degradation
- **Evaluation Framework**: Measures response quality and relevance using DSPy's evaluation capabilities

## Architecture

```ascii
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Interface │────▶│  Query Router   │────▶│  RAG Pipeline   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │  ▲                     │
                               ▼  │                     ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Tools (Calc,  │     │  Vector Store   │
                        │   Web Search)   │     │   (ChromaDB)    │
                        └─────────────────┘     └─────────────────┘
                               │  ▲                     │
                               ▼  │                     ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   LLM Provider  │◀───▶│  DSPy Modules   │
                        │ (OpenAI/Claude) │     │    & Metrics    │
                        └─────────────────┘     └─────────────────┘
```

## Setup & Installation

### Local Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/LLM_research_assistant.git
cd LLM_research_assistant
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"  # If using Claude
export SERPER_API_KEY="your-api-key"     # If using Serper.dev for web search
```

4. Prepare your data

```bash
python src/vectorstore_builder.py --data_path data/raw --output_path data/vector_stores
```

### Docker Installation

1. Build and run with Docker Compose

```bash
docker-compose up -d
```

## Usage

### Basic Usage

```python
from src.agent import ResearchAssistant
from src.llm_providers import OpenAILLM
from src.rag_pipeline import RAGPipeline
from src.tools.web_search import WebSearch
from src.tools.calculator import Calculator

# Initialize components
llm = OpenAILLM(model_name="gpt-4o")
rag = RAGPipeline()
rag.initialize()
retriever = rag.get_retriever()

# Set up tools
tools = {
    "calculator": Calculator(),
    "web_search": WebSearch(api_key="your-search-api-key", search_engine="serper")
}

# Create and use the assistant
assistant = ResearchAssistant(llm_provider=llm, retriever=retriever, tools=tools)
response = assistant.process_query("What was the GDP growth rate in the US last quarter?")
print(response["response"])
```

### API Usage

Start the API server:

```bash
uvicorn api_gateway:app --host 0.0.0.0 --port 8000
```

Send a query via HTTP:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the benefits of RAG over traditional LLM approaches?"}'
```

### Multi-Modal Usage

```python
from src.agent import MultiModalAssistant
from src.llm_providers import AnthropicLLM

# Initialize multi-modal assistant
llm = AnthropicLLM(model_name="claude-3-opus-20240229")
assistant = MultiModalAssistant(llm_provider=llm, retriever=retriever, tools=tools)

# Process query with image
response = assistant.process_query(
    "What can you tell me about this graph?", 
    images=["path/to/image.png"]
)
```

## Project Structure

```
llm-research-assistant/
├── api_gateway.py         # FastAPI server for the assistant
├── data/
│   ├── raw/                # Original dataset files
│   ├── processed/          # Cleaned CSV files
│   └── vector_stores/      # ChromaDB vector stores
├── docker-compose.yml     # Docker configuration
├── Dockerfile             # Docker build instructions
├── prompts/
│   └── query_classification_prompt_template.txt  # LLM prompts
├── src/
│   ├── agent.py            # Main assistant logic
│   ├── llm_providers.py    # LLM abstraction layer
│   ├── rag_pipeline.py     # Document retrieval system
│   ├── router.py           # Query routing logic
│   ├── tools/              # External tool integrations
│   │   ├── calculator.py   # Math calculation tool
│   │   └── web_search.py   # Web search with multiple engines
│   ├── dspy_modules/       # DSPy components
│   │   ├── evaluators.py   # Evaluation metrics
│   │   └── signatures.py   # DSPy signatures
│   └── vectorstore_builder.py  # Indexing utility
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test fixtures
├── config.yaml            # Configuration file
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

## Advanced Features

### Hybrid Retrieval

The system combines dense vector similarity search with sparse BM25 retrieval for better document retrieval:

```python
from src.retrieval import HybridRetriever

# Create a hybrid retriever
hybrid_retriever = HybridRetriever(
    vector_store=chroma_db,
    sparse_weight=0.3,
    dense_weight=0.7
)

# Use in assistant
assistant = ResearchAssistant(llm_provider=llm, retriever=hybrid_retriever)
```

### Caching

Enable result caching to improve performance:

```python
from src.utils.caching import ResultCache, cached

# Initialize cache
cache = ResultCache(redis_url="redis://localhost:6379/0")

# Apply caching to expensive operations
@cached(cache)
def get_embeddings(text):
    # Expensive embedding computation
    return embeddings
```

## Error Handling

The system implements robust error handling with custom exceptions:

```python
try:
    response = assistant.process_query("Complex query")
except LLMProviderError as e:
    # Handle LLM-specific errors
    fallback_response = "I'm having trouble connecting to my knowledge base"
except RAGPipelineError as e:
    # Handle retrieval errors
    fallback_response = "I couldn't retrieve the necessary information"
except ToolExecutionError as e:
    # Handle tool execution errors
    fallback_response = "I encountered an issue with the requested operation"
```

## Requirements

- Python 3.8+
- LangChain
- DSPy
- ChromaDB
- OpenAI or Anthropic API access
- Redis (optional, for caching)

## Evaluation

The system uses DSPy's evaluation framework to assess:

- Answer correctness
- Context relevance
- Reasoning quality
- Hallucination detection

## Contributing

We welcome contributions to improve the research assistant! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [DSPy](https://github.com/stanfordnlp/dspy)
- Vector storage provided by [ChromaDB](https://github.com/chroma-core/chroma)