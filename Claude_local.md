# LLM-Powered Research Assistant

This AI research assistant project uses Retrieval-Augmented Generation (RAG) to answer user queries with improved accuracy and capabilities.

## Current Implementation

### Features
- Smart query routing (decides whether to answer directly, retrieve context, or use tools)
- RAG pipeline for document retrieval
- Multi-step reasoning with DSPy
- Tool integration (calculator, web search)
- Evaluation framework using DSPy metrics

### Architecture
```
User Interface → Query Router → RAG Pipeline
                      ↓               ↓
                Tools (Calc,    Vector Store
                Web Search)     (ChromaDB)
                      ↓               ↓
                LLM Provider ← → DSPy Modules
                (OpenAI/Claude)    & Metrics
```

### Code Structure
- `src/agent.py` - Main assistant logic
- `src/llm_providers.py` - LLM abstraction
- `src/rag_pipeline.py` - Document retrieval
- `src/router.py` - Query routing
- `src/tools/` - External tools
- `src/dspy_modules/` - DSPy components

### Current Usage
```python
from src.agent import ResearchAssistant
from src.llm_providers import OpenAILLM
from src.rag_pipeline import RAGPipeline

llm = OpenAILLM(model_name="gpt4o")
rag = RAGPipeline()
rag.initialize()
retriever = rag.get_retriever()

assistant = ResearchAssistant(llm_provider=llm, retriever=retriever)
response = assistant.process_query("How do I fix Wi-Fi connection issues?")
```

## Recommended Improvements

### 1. Code Architecture and Design Patterns

#### Dependency Injection Refinement
- Implement a proper dependency injection container (e.g., `python-dependency-injector`) to manage service lifecycles
- Create a configuration system for easier environment management (dev/prod/test)

```python
# Example improved initialization with DI container
from dependency_injector import containers, providers
from src.agent import ResearchAssistant
from src.llm_providers import OpenAILLM, AnthropicLLM

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # LLM providers with factory pattern
    llm = providers.Factory(
        lambda provider_name, **kwargs: 
            OpenAILLM(**kwargs) if provider_name == "openai" 
            else AnthropicLLM(**kwargs)
    )
    
    # RAG pipeline
    rag_pipeline = providers.Singleton(
        RAGPipeline,
        persist_directory=config.vectorstore.directory,
        embedding_model=config.embeddings.model
    )
    
    # Tool registry with dynamic loading
    tool_registry = providers.Dict({
        "calculator": providers.Factory(Calculator),
        "web_search": providers.Factory(
            WebSearch,
            api_key=config.tools.web_search.api_key
        )
    })
    
    # Main assistant
    assistant = providers.Singleton(
        ResearchAssistant,
        llm_provider=llm,
        retriever=lambda: rag_pipeline().get_retriever(),
        tools=tool_registry
    )

# Usage
container = Container()
container.config.from_yaml("config.yaml")
assistant = container.assistant()
```

#### Command Pattern for Tools
- Standardize tool interface with a clear Command pattern
- Implement uniform input/output handling for all tools

```python
# Improved tool interface
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ToolInput:
    query: str
    parameters: Dict[str, Any] = None

@dataclass
class ToolOutput:
    result: str
    metadata: Dict[str, Any] = None
    success: bool = True
    error_message: Optional[str] = None

class Tool(ABC):
    @abstractmethod
    def execute(self, input_data: ToolInput) -> ToolOutput:
        pass
        
    @property
    def name(self) -> str:
        return self.__class__.__name__
        
    @property
    def description(self) -> str:
        return self.__doc__ or "No description available"
```

### 2. Web Search Implementation

The current web search implementation is a placeholder with mock results. Here's an improved version using a real search API:

```python
# Improved web search implementation
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str
    published_date: Optional[str] = None

class WebSearch(Tool):
    """Tool for performing web searches using various search engines"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        search_engine: str = "serpapi",
        result_count: int = 5,
        timeout: int = 10,
        safe_search: bool = True
    ):
        self.api_key = api_key
        self.search_engine = search_engine
        self.result_count = result_count
        self.timeout = timeout
        self.safe_search = safe_search
        
        # Set up the appropriate client based on search engine
        if search_engine == "serpapi":
            self._setup_serpapi()
        elif search_engine == "serper":
            self._setup_serper()
        elif search_engine == "bing":
            self._setup_bing()
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")
    
    def _setup_serpapi(self):
        # Setup for SerpAPI
        try:
            from serpapi import GoogleSearch
            self.client = GoogleSearch
        except ImportError:
            raise ImportError("Please install serpapi: pip install google-search-results")
    
    def _setup_serper(self):
        # Serper.dev setup
        self.serper_url = "https://serper.dev/search"
        self.serper_headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def _setup_bing(self):
        # Bing setup
        self.bing_url = "https://api.bing.microsoft.com/v7.0/search"
        self.bing_headers = {
            'Ocp-Apim-Subscription-Key': self.api_key
        }
    
    def execute(self, input_data: ToolInput) -> ToolOutput:
        """
        Execute a web search query
        
        Args:
            input_data: ToolInput containing the search query
            
        Returns:
            ToolOutput with search results
        """
        query = input_data.query
        parameters = input_data.parameters or {}
        
        try:
            if self.search_engine == "serpapi":
                results = self._search_serpapi(query, parameters)
            elif self.search_engine == "serper":
                results = self._search_serper(query, parameters)
            elif self.search_engine == "bing":
                results = self._search_bing(query, parameters)
            else:
                return ToolOutput(
                    result="Error: Unsupported search engine",
                    success=False,
                    error_message=f"Search engine {self.search_engine} not supported"
                )
                
            # Format results and return
            formatted_results = self._format_results(results)
            return ToolOutput(
                result=formatted_results,
                metadata={"raw_results": results, "query": query},
                success=True
            )
        except Exception as e:
            return ToolOutput(
                result="Error performing web search",
                success=False,
                error_message=str(e)
            )
    
    def _search_serpapi(self, query: str, parameters: Dict[str, Any]) -> List[SearchResult]:
        """Search using SerpAPI"""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": self.result_count,
            "safe": "active" if self.safe_search else "off"
        }
        params.update(parameters)
        
        search = self.client(params)
        results = search.get_dict()
        
        # Parse results
        search_results = []
        if "organic_results" in results:
            for result in results["organic_results"][:self.result_count]:
                search_results.append(SearchResult(
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    url=result.get("link", ""),
                    source="google",
                    published_date=result.get("date", "")
                ))
        return search_results
        
    def _search_serper(self, query: str, parameters: Dict[str, Any]) -> List[SearchResult]:
        """Search using Serper.dev"""
        payload = {
            "q": query,
            "num": self.result_count
        }
        payload.update(parameters)
        
        response = requests.post(
            self.serper_url,
            headers=self.serper_headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # Parse results
        search_results = []
        if "organic" in data:
            for result in data["organic"][:self.result_count]:
                search_results.append(SearchResult(
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    url=result.get("link", ""),
                    source="serper",
                    published_date=result.get("date", "")
                ))
        return search_results
    
    def _search_bing(self, query: str, parameters: Dict[str, Any]) -> List[SearchResult]:
        """Search using Bing API"""
        params = {
            "q": query,
            "count": self.result_count,
            "safeSearch": "strict" if self.safe_search else "off"
        }
        params.update(parameters)
        
        response = requests.get(
            self.bing_url,
            headers=self.bing_headers,
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # Parse results
        search_results = []
        if "webPages" in data and "value" in data["webPages"]:
            for result in data["webPages"]["value"][:self.result_count]:
                search_results.append(SearchResult(
                    title=result.get("name", ""),
                    snippet=result.get("snippet", ""),
                    url=result.get("url", ""),
                    source="bing"
                ))
        return search_results
    
    def _format_results(self, results: List[SearchResult]) -> str:
        """Format the search results into a readable string"""
        if not results:
            return "No search results found."
            
        formatted = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.title}\n"
            formatted += f"   {result.snippet}\n"
            formatted += f"   URL: {result.url}\n"
            if result.published_date:
                formatted += f"   Published: {result.published_date}\n"
            formatted += f"   Source: {result.source}\n\n"
        return formatted
```

### 3. Error Handling and Robustness

#### Centralized Error Management
- Implement a custom exception hierarchy
- Add structured logging and monitoring
- Implement graceful degradation when components fail

```python
# Exception hierarchy
class ResearchAssistantError(Exception):
    """Base exception for all research assistant errors"""
    pass

class LLMProviderError(ResearchAssistantError):
    """Errors related to LLM providers"""
    pass

class RAGPipelineError(ResearchAssistantError):
    """Errors related to the RAG pipeline"""
    pass

class ToolExecutionError(ResearchAssistantError):
    """Errors related to tool execution"""
    pass

# Error monitoring and reporting service
class ErrorMonitor:
    def __init__(self, monitoring_service=None):
        self.service = monitoring_service
        
    def capture_exception(self, exception, context=None):
        """Capture an exception and its context"""
        # Log the error
        logger.error(f"Error: {str(exception)}", exc_info=True, extra=context or {})
        
        # Send to monitoring service if available
        if self.service:
            self.service.report_error(exception, context)
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.capture_exception(exc_val)
            return False  # Re-raise the exception
```

#### Retry Logic
- Add retry mechanisms for external services
- Implement circuit breakers for external dependencies

```python
# Retry decorator
import time
from functools import wraps

def retry(exceptions, tries=4, delay=3, backoff=2):
    """
    Retry decorator with exponential backoff
    
    Args:
        exceptions: The exceptions to catch and retry
        tries: Number of times to try (not retry) before giving up
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier e.g. value of 2 will double the delay each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retrying {func.__name__} in {mdelay}s: {str(e)}")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### 4. RAG Pipeline Optimization

#### Advanced Document Processing
- Implement smarter chunking strategies
- Add metadata extraction for improved retrieval
- Support more document types (PDF, HTML, etc.)

```python
# Improved document processing
from langchain.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EnhancedDocumentProcessor:
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        metadata_extractors=None
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.metadata_extractors = metadata_extractors or []
        
    def load_document(self, file_path):
        """Load a document based on its file extension"""
        extension = file_path.split(".")[-1].lower()
        
        if extension == "pdf":
            loader = PyPDFLoader(file_path)
        elif extension in ["html", "htm"]:
            loader = BSHTMLLoader(file_path)
        elif extension in ["txt", "md"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
            
        return loader.load()
        
    def extract_metadata(self, document):
        """Extract metadata from document"""
        for extractor in self.metadata_extractors:
            document.metadata.update(extractor.extract(document))
        return document
        
    def process_document(self, document):
        """Process a document by extracting metadata and splitting into chunks"""
        # Extract metadata
        document = self.extract_metadata(document)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([document])
        
        # Ensure each chunk inherits metadata
        for chunk in chunks:
            chunk.metadata.update(document.metadata)
            
        return chunks
```

#### Hybrid Search and Re-ranking
- Implement hybrid search (sparse + dense vectors)
- Add re-ranking for improved result quality

```python
# Hybrid retrieval system
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

class HybridRetriever:
    def __init__(
        self,
        vector_store,
        sparse_weight=0.5,
        dense_weight=0.5,
        reranker=None
    ):
        # Dense retriever from vector store
        self.dense_retriever = vector_store.as_retriever()
        
        # Sparse retriever using BM25
        self.sparse_retriever = BM25Retriever.from_documents(
            vector_store.get_all_documents()
        )
        
        # Create ensemble
        self.retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[dense_weight, sparse_weight]
        )
        
        # Optional reranker
        self.reranker = reranker
        
    def get_relevant_documents(self, query, k=5):
        """Get relevant documents using hybrid retrieval"""
        docs = self.retriever.get_relevant_documents(query)
        
        # Apply reranking if available
        if self.reranker:
            docs = self.reranker.rerank(query, docs)
            
        return docs[:k]
```

### 5. Multi-Modal Support

Add support for processing and responding to images and other media:

```python
# Multi-modal support
from typing import List, Union, Optional
from PIL import Image

class MultiModalProcessor:
    def __init__(self, image_model=None):
        self.image_model = image_model
        
    def process_image(self, image_path: str) -> str:
        """Process an image and return a textual description"""
        try:
            image = Image.open(image_path)
            
            # Use the image model to generate a description
            if self.image_model:
                description = self.image_model.describe_image(image)
                return description
            else:
                return "Image processing model not available"
        except Exception as e:
            return f"Error processing image: {str(e)}"
            
class MultiModalAssistant(ResearchAssistant):
    def __init__(
        self,
        llm_provider,
        retriever,
        tools=None,
        image_processor=None,
        **kwargs
    ):
        super().__init__(llm_provider, retriever, tools, **kwargs)
        self.image_processor = image_processor
        
    def process_query(self, query, images=None):
        """Process a query with optional images"""
        if images:
            # Process images and add descriptions to the query
            image_descriptions = []
            for img_path in images:
                description = self.image_processor.process_image(img_path)
                image_descriptions.append(description)
                
            # Augment query with image descriptions
            augmented_query = f"{query}\n\nImage descriptions:\n"
            for i, desc in enumerate(image_descriptions, 1):
                augmented_query += f"Image {i}: {desc}\n"
                
            # Process the augmented query
            return super().process_query(augmented_query)
        else:
            return super().process_query(query)
```

### 6. Testing Strategy

#### Unit Testing Framework
- Implement comprehensive unit tests
- Add test fixtures and mocks for external dependencies

```python
# tests/test_agent.py
import pytest
from unittest.mock import Mock, patch
from src.agent import ResearchAssistant
from src.llm_providers import LLMProvider

@pytest.fixture
def mock_llm():
    """Create a mock LLM provider"""
    mock = Mock(spec=LLMProvider)
    mock.generate_response.return_value = "This is a mock response"
    return mock
    
@pytest.fixture
def mock_retriever():
    """Create a mock retriever"""
    mock = Mock()
    mock.get_relevant_documents.return_value = [
        Mock(page_content="Relevant document content")
    ]
    return mock
    
@pytest.fixture
def mock_tool():
    """Create a mock tool"""
    mock = Mock()
    mock.execute.return_value = "Tool execution result"
    return mock
    
@pytest.fixture
def assistant(mock_llm, mock_retriever):
    """Create a research assistant with mocked dependencies"""
    tools = {"mock_tool": mock_tool()}
    return ResearchAssistant(
        llm_provider=mock_llm,
        retriever=mock_retriever,
        tools=tools
    )

def test_direct_knowledge_handling(assistant, mock_llm):
    """Test handling of direct knowledge queries"""
    # Arrange
    classification = {"type": "direct_knowledge"}
    
    # Act
    result = assistant.handle_direct_knowledge("test query", classification)
    
    # Assert
    assert result == "This is a mock response"
    mock_llm.generate_response.assert_called_once()
    
def test_research_handling(assistant, mock_retriever, mock_llm):
    """Test handling of research queries"""
    # Arrange
    classification = {"type": "research_needed"}
    
    # Act
    result = assistant.handle_research("test query", classification)
    
    # Assert
    assert result == "This is a mock response"
    mock_retriever.get_relevant_documents.assert_called_once_with("test query")
    mock_llm.generate_response.assert_called_once()
```

#### Integration Testing
- Add integration tests for key workflows
- Implement end-to-end testing with real components

```python
# tests/integration/test_end_to_end.py
import pytest
from src.agent import ResearchAssistant
from src.llm_providers import OpenAILLM
from src.rag_pipeline import RAGPipeline

@pytest.mark.integration
def test_end_to_end_query_processing():
    """Test the full query processing pipeline with real components"""
    # Setup
    llm = OpenAILLM(model_name="gpt-3.5-turbo")
    rag = RAGPipeline(persist_directory="test_vectorstore")
    
    # Initialize RAG with test data
    rag.initialize({
        "files": ["tests/data/test_document.txt"]
    })
    
    # Create assistant
    assistant = ResearchAssistant(
        llm_provider=llm,
        retriever=rag.get_retriever()
    )
    
    # Process a test query
    response = assistant.process_query("What is in the test document?")
    
    # Verify response contains expected content
    assert response is not None
    assert isinstance(response, dict)
    assert "response" in response
    assert len(response["response"]) > 0
```

### 7. Deployment and Scaling

#### Containerization
- Add Docker support for easy deployment
- Include docker-compose for local development

```Dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run as non-root user
RUN useradd -m appuser
USER appuser

# Default command
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3'

services:
  assistant:
    build: .
    volumes:
      - .:/app
    env_file:
      - .env
    ports:
      - "8000:8000"
      
  vectorstore:
    image: chromadb/chroma
    volumes:
      - ./data/chromadb:/chroma/data
    ports:
      - "8001:8000"
```

#### API Gateway
- Create a REST API for the assistant

```python
# api_gateway.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from src.agent import ResearchAssistant
from src.llm_providers import OpenAILLM, AnthropicLLM
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="Research Assistant API")

# Load components
llm = OpenAILLM(model_name="gpt-4o")
rag = RAGPipeline()
rag.initialize()
retriever = rag.get_retriever()

assistant = ResearchAssistant(
    llm_provider=llm,
    retriever=retriever
)

class QueryRequest(BaseModel):
    query: str
    tools: List[str] = []
    
class QueryResponse(BaseModel):
    response: str
    query_type: str
    metadata: Dict[str, Any] = {}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query and return the result"""
    try:
        result = assistant.process_query(request.query)
        return {
            "response": result["response"],
            "query_type": result["query_type"],
            "metadata": {
                "classification": result.get("classification", {}),
                "tools_used": result.get("tools_used", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Performance Optimization
- Implement caching for query results and embeddings
- Add asynchronous processing for heavy operations

```python
# Caching implementation
import functools
import hashlib
import json
from typing import Any, Callable, Dict, Optional
import redis

class ResultCache:
    def __init__(self, redis_url=None, ttl=3600):
        """Initialize the cache with Redis or in-memory dict"""
        self.ttl = ttl
        if redis_url:
            self.redis = redis.from_url(redis_url)
            self.use_redis = True
        else:
            self.cache = {}
            self.use_redis = False
    
    def _get_key(self, func_name: str, args: tuple, kwargs: Dict[str, Any]) -> str:
        """Generate a unique cache key"""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache"""
        if self.use_redis:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        else:
            return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache"""
        if self.use_redis:
            self.redis.setex(key, self.ttl, json.dumps(value))
        else:
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache"""
        if self.use_redis:
            self.redis.flushdb()
        else:
            self.cache.clear()

def cached(cache: ResultCache):
    """Decorator to cache function results"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache._get_key(func.__name__, args, kwargs)
            
            # Check if result is in cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator
```

## Future Enhancements

### 1. Advanced RAG Techniques
- Implement query rewriting for better retrieval
- Add query-focused summarization
- Implement parent-child retrieval (hierarchical chunking)

### 2. Explainability and Transparency
- Add citation tracking for claims made by the LLM
- Implement confidence scoring for responses
- Add uncertainty estimation

### 3. User Feedback Integration
- Collect and use feedback for response quality
- Implement active learning for query handling improvement
- Add reinforcement learning from human feedback (RLHF)

### 4. Advanced Tool Management
- Implement plugin architecture for easy tool addition
- Add parameter extraction for tools from natural language
- Support tool composition and chaining

### 5. Domain Specialization
- Add domain-specific retrieval strategies
- Implement custom embeddings for specific domains
- Create specialized response templates for different contexts

## Getting Started with Improvements

To begin implementing these improvements:

1. Refactor the code architecture using the DI container approach
2. Replace the mock web search with a real implementation
3. Enhance error handling with the exception hierarchy
4. Add hybrid search to the RAG pipeline
5. Implement unit and integration tests
6. Set up Docker for deployment

The resulting system will be more robust, maintainable, and ready for production use.