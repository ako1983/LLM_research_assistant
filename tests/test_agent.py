# tests/test_agent.py
import unittest
import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ResearchAssistant
from src.llm_providers import OpenAILLM
from src.tools.calculator import Calculator
from src.tools.web_search import WebSearch
from langchain.schema import Document

class MockRetriever:
    """Mock retriever for testing"""
    def get_relevant_documents(self, query):
        return [
            Document(page_content="This is a test document that contains information related to the query.", metadata={"source": "test"}),
            Document(page_content="Here is another test document with different information.", metadata={"source": "test2"})
        ]

class TestResearchAssistant(unittest.TestCase):
    """Test cases for the ResearchAssistant class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use a mock LLM provider to avoid actual API calls during tests
        class MockLLM(OpenAILLM):
            def generate_response(self, prompt, **kwargs):
                return f"Mock response to: {prompt[:50]}..."
        
        self.mock_llm = MockLLM()
        self.mock_retriever = MockRetriever()
        self.tools = {
            "calculator": Calculator(),
            "web_search": WebSearch()
        }
        
        # Initialize the agent with mocks
        self.agent = ResearchAssistant(
            llm_provider=self.mock_llm,
            retriever=self.mock_retriever,
            tools=self.tools
        )
    
    def test_direct_knowledge_query(self):
        """Test handling of direct knowledge queries"""
        result = self.agent.handle_direct_knowledge(
            "What is the capital of France?", 
            {"type": "direct_knowledge"}
        )
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, str))
    
    def test_research_query(self):
        """Test handling of research queries"""
        result = self.agent.handle_research(
            "Tell me about quantum computing", 
            {"type": "research_needed"}
        )
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, str))
    
    def test_tool_query(self):
        """Test handling of tool-requiring queries"""
        result = self.agent.handle_tool_call(
            "Calculate 2 + 2", 
            {"type": "tool_required", "tool_name": "calculator"}
        )
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, str))

if __name__ == "__main__":
    unittest.main()