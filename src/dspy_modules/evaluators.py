# In src/dspy_modules/evaluators.py
import dspy

class ResponseAccuracy(dspy.Metric):
    """Evaluates the accuracy of responses against reference answers."""
    
    def __call__(self, example, pred, trace=None):
        # For this example, we'll use a simulated accuracy score
        # In a real system, you would compare against gold answers
        # or use more sophisticated evaluation metrics
        
        reference_answer = example.get("reference", "")
        predicted_answer = pred.get("answer", "")
        
        # Simple string similarity (not a great metric, but simple for demo)
        overlap = len(set(reference_answer.split()) & set(predicted_answer.split()))
        total = len(set(reference_answer.split()) | set(predicted_answer.split()))
        
        if total == 0:
            return 0
        
        similarity = overlap / total
        return similarity

class RetrievalRelevance(dspy.Metric):
    """Evaluates the relevance of retrieved documents to the query."""
    
    def __call__(self, example, pred, trace=None):
        # Again, this would be more sophisticated in a real system
        query = example.get("query", "")
        retrieved_docs = pred.get("retrieved_docs", [])
        
        # Simulate relevance scores
        # In a real system, this could use embedding similarity
        relevance_scores = []
        for doc in retrieved_docs:
            # Count query term occurrence (very simplistic)
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            relevance_scores.append(min(1.0, overlap / max(1, len(query_terms))))
            
        if not relevance_scores:
            return 0
            
        # Return average relevance
        return sum(relevance_scores) / len(relevance_scores)