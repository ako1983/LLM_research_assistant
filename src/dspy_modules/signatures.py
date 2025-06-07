# In src/dspy_modules/signatures.py
import dspy

class QueryClassifier(dspy.Signature):
    """Classify the type of query from a user."""
    query = dspy.InputField(desc="The user's query")
    query_type = dspy.OutputField(desc="One of: direct_knowledge, research_needed, tool_required")
    reasoning = dspy.OutputField(desc="Reasoning for the classification")

class ResearchAgent(dspy.Signature):
    """Research agent that can retrieve context and answer questions."""
    query = dspy.InputField(desc="The user's query")
    thoughts = dspy.OutputField(desc="Step-by-step thinking about how to answer the query")
    needs_research = dspy.OutputField(desc="Whether external information is needed (yes/no)")
    needs_tool = dspy.OutputField(desc="Whether a tool is needed (yes/no and which tool)")
    answer = dspy.OutputField(desc="The final answer to the query")

class MultiStepReasoner(dspy.Module):
    """Module that implements multi-step reasoning."""
    
    def __init__(self, retriever=None, tools=None):
        super().__init__()
        self.retriever = retriever
        self.tools = tools or {}
        self.classifier = dspy.Predict(QueryClassifier)
        self.researcher = dspy.ChainOfThought(ResearchAgent)
        
    def forward(self, query):
        # Step 1: Classify the query
        classification = self.classifier(query=query)
        
        # Step 2: Think through the reasoning process
        research_result = self.researcher(query=query)
        
        # Step 3: Determine if we need to retrieve documents
        if research_result.needs_research == "yes":
            if self.retriever:
                docs = self.retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
                # Use the context to generate a better answer
                # This would require a custom DSPy module that integrates context
            else:
                # No retriever available, proceed with base answer
                pass
        
        # Step 4: Determine if we need to use tools
        if research_result.needs_tool.startswith("yes"):
            tool_name = research_result.needs_tool.split("yes, ")[1]
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    tool_result = tool.execute(query)
                    # Use the tool result to enhance the answer
                    # This would require a custom DSPy module that integrates tool results
                except Exception:
                    # Tool execution failed, proceed with base answer
                    pass
        
        # Return the final result
        return {
            "query": query,
            "classification": classification,
            "thoughts": research_result.thoughts,
            "answer": research_result.answer
        }