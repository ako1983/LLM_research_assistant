# src/agent.py
from typing import Dict, List, Any, Optional
import logging
from .llm_providers import LLMProvider
from .router import QueryRouter
from .dspy_modules.signatures import MultiStepReasoner
import dspy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchAssistant:
    def __init__(
        self, 
        llm_provider: LLMProvider,
        retriever,  # Your ChromaDB retriever
        tools: Dict[str, Any] = None,
        prompt_dir: str = "prompts"
    ):
        self.llm = llm_provider
        self.retriever = retriever
        self.tools = tools or {}
        self.router = QueryRouter(
            llm_provider=llm_provider,
            prompt_path=f"{prompt_dir}/query_classification_prompt_template.txt"
        )
        
        # Initialize DSPy components
        try:
            if isinstance(llm_provider.__class__.__name__, "OpenAILLM"):
                self.dspy_llm = dspy.OpenAI(model=llm_provider.model_name)
            else:
                self.dspy_llm = dspy.Anthropic(model=llm_provider.model_name)
            dspy.settings.configure(lm=self.dspy_llm)
            self.reasoner = MultiStepReasoner(retriever=retriever, tools=tools)
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy components: {e}")
            self.reasoner = None
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for processing user queries.
        
        Args:
            query: The user's question or request
            
        Returns:
            A dictionary containing the response and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # 1. Try using DSPy reasoner if available
        if self.reasoner:
            try:
                logger.info("Attempting to use DSPy multi-step reasoner")
                dspy_result = self.reasoner(query)
                return {
                    "query": query,
                    "response": dspy_result["answer"],
                    "query_type": "dspy_reasoner",
                    "reasoning": dspy_result.get("thoughts", ""),
                    "classification": dspy_result.get("classification", {})
                }
            except Exception as e:
                logger.warning(f"DSPy reasoning failed, falling back to standard pipeline: {e}")
        
        # 2. Classify the query using the router
        classification = self.router.classify_query(
            user_query=query,
            knowledge_base="Research papers and technical documents",
            available_functions=str(list(self.tools.keys()))
        )
        
        logger.info(f"Query classified as: {classification['type']}")
        
        # 3. Route to appropriate handler
        query_type = classification.get("type", "unknown")
        
        if query_type == "direct_knowledge":
            response = self.handle_direct_knowledge(query, classification)
        elif query_type == "research_needed":
            response = self.handle_research(query, classification)
        elif query_type == "tool_required":
            response = self.handle_tool_call(query, classification)
        else:
            # Default to research if classification fails
            logger.warning(f"Unknown query type: {query_type}, defaulting to research handler")
            response = self.handle_research(query, classification)
            
        # 4. Return response with metadata
        return {
            "query": query,
            "response": response,
            "query_type": query_type,
            "classification": classification
        }
        
    def handle_direct_knowledge(self, query: str, classification: Dict[str, Any]) -> str:
        """Handle queries that can be answered directly from LLM knowledge."""
        logger.info("Handling direct knowledge query")
        
        # If the classifier already generated an answer, use it
        if classification.get("answer") and len(classification["answer"]) > 10:
            return classification["answer"]
        
        # Otherwise, generate a new response
        prompt = f"""Answer the following question using your existing knowledge:
        
        Question: {query}
        
        Provide a comprehensive, accurate response based on what you know.
        Use step-by-step reasoning to ensure accuracy.
        """
        return self.llm.generate_response(prompt)
        
    def handle_research(self, query: str, classification: Dict[str, Any]) -> str:
        """Handle queries that require retrieving documents."""
        logger.info("Handling research query")
        
        # Get relevant documents
        try:
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Format documents for context
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                for i, doc in enumerate(docs)])
            
            # Create augmented prompt with reasoning
            prompt = f"""Answer the following question using the provided context and your knowledge:
            
            Question: {query}
            
            Context:
            {context}
            
            Instructions:
            1. First analyze what information in the provided context is relevant to the question.
            2. If the context doesn't contain all necessary information, supplement with your knowledge.
            3. Use step-by-step reasoning to arrive at your answer.
            4. Provide citations to specific documents when using information from the context.
            5. If the context doesn't have relevant information, clearly state that.
            
            Your response:
            """
            return self.llm.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error in research handler: {e}")
            # Fallback to direct knowledge
            return f"I encountered an issue retrieving documents. Let me answer based on my knowledge instead.\n\n" + self.handle_direct_knowledge(query, classification)
        
    def handle_tool_call(self, query: str, classification: Dict[str, Any]) -> str:
        """Handle queries that require calling external tools."""
        logger.info("Handling tool query")
        
        # Identify which tool to use from the classification
        tool_name = classification.get("tool_name")
        
        # If no tool was identified in classification, try to find one
        if not tool_name:
            for tool in self.tools.keys():
                if tool.lower() in query.lower() or tool.lower() in classification.get("action", "").lower():
                    tool_name = tool
                    break
                    
        logger.info(f"Selected tool: {tool_name}")
                
        if not tool_name or tool_name not in self.tools:
            # Default to the first tool or return research results
            if self.tools:
                tool_name = list(self.tools.keys())[0]
                logger.warning(f"No specific tool identified, defaulting to: {tool_name}")
            else:
                logger.warning("No tools available, falling back to research")
                return self.handle_research(query, classification)
                
        # Execute the tool
        tool = self.tools[tool_name]
        try:
            logger.info(f"Executing tool: {tool_name}")
            tool_result = tool.execute(query)
            
            # Create response with tool result and reasoning
            prompt = f"""The user asked: {query}
            
            I used the {tool_name} tool and got this result:
            {tool_result}
            
            Based on this information, please provide:
            1. An explanation of how this result answers the user's question
            2. Any additional context or information that would be helpful
            3. A clear, concise answer that incorporates the tool's output
            
            Use step-by-step reasoning in your response.
            """
            return self.llm.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            # Fallback to research if tool execution fails
            return f"I tried to use the {tool_name} tool but encountered an error. Let me research this instead.\n\n" + self.handle_research(query, classification)