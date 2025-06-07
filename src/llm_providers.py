# src/llm_providers.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Abstract LLM interface
class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM given a prompt."""
        pass

# Concrete implementation for Anthropic
class AnthropicLLM(LLMProvider):
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(**kwargs)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install the Anthropic Python package: pip install anthropic")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", 0.1),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

### or OpenAI ###
class OpenAILLM(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        try:
            from openai import OpenAI
            self.client = OpenAI(**kwargs)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install the OpenAI Python package: pip install openai")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Unified prompt template class
class PromptTemplate:
    @classmethod
    def from_file(cls, file_path: str, input_variables: list):
        with open(file_path, 'r') as file:
            template = file.read()
        return cls(template, input_variables)
    
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        """Format the template with the provided variables."""
        formatted_prompt = self.template
        for var in self.input_variables:
            if var in kwargs:
                formatted_prompt = formatted_prompt.replace(f"{{{var}}}", str(kwargs[var]))
        return formatted_prompt