# src/router.py
import re
from typing import Dict, Any, Optional
from .llm_providers import LLMProvider, PromptTemplate

#  QueryRouter class
class QueryRouter:
    def __init__(self, llm_provider: LLMProvider, prompt_path: str):
        self.llm = llm_provider
        try:
            self.prompt_template = PromptTemplate.from_file(
                prompt_path,
                input_variables=["KNOWLEDGE_BASE", "AVAILABLE_FUNCTIONS", "USER_QUERY"]
            )
        except FileNotFoundError:
            # Fallback to query_classification_prompt_template.txt
            import os
            fallback_path = os.path.join("prompts", "query_classification_prompt_template.txt")
            self.prompt_template = PromptTemplate.from_file(
                fallback_path,
                input_variables=["KNOWLEDGE_BASE", "AVAILABLE_FUNCTIONS", "USER_QUERY"]
            )
    
    def classify_query(self, user_query: str, knowledge_base: str = "", available_functions: str = "") -> Dict[str, Any]:
        """
        Classify the query type and determine how to process it.
        
        Args:
            user_query: The user's query to be classified
            knowledge_base: Description of the knowledge base available
            available_functions: List of available tools/functions
            
        Returns:
            A dictionary containing the classification results
        """
        formatted_prompt = self.prompt_template.format(
            KNOWLEDGE_BASE=knowledge_base,
            AVAILABLE_FUNCTIONS=available_functions,
            USER_QUERY=user_query
        )
        
        response = self.llm.generate_response(formatted_prompt)
        
        # Parse the response to extract classification
        classification = self._parse_classification(response)
        return classification
    
    def _parse_classification(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured information from XML-like tags.
        Expected format:
        <decision>...</decision>
        <reasoning>...</reasoning>
        <action>...</action>
        <answer>...</answer>
        
        Returns:
            A dictionary with keys for type, reasoning, action, and answer
        """
        result = {
            "type": "unknown",
            "reasoning": "",
            "action": "",
            "answer": ""
        }
        
        # Extract decision (maps to type)
        decision_match = re.search(r'<decision>(.*?)</decision>', response, re.DOTALL)
        if decision_match:
            decision = decision_match.group(1).strip().lower()
            if "direct answer" in decision:
                result["type"] = "direct_knowledge"
            elif "need for more context" in decision:
                result["type"] = "research_needed"
            elif "function call" in decision:
                result["type"] = "tool_required"
        
        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        if action_match:
            result["action"] = action_match.group(1).strip()
            
            # Try to extract tool name from action if it's a tool call
            if result["type"] == "tool_required":
                tool_match = re.search(r'call (?:the|function)?\s*["\']?([a-zA-Z0-9_]+)["\']?', result["action"], re.IGNORECASE)
                if tool_match:
                    result["tool_name"] = tool_match.group(1)
        
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        
        return result