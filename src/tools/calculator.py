# src/tools/calculator.py
import re
import math
from typing import Dict, Any, Optional

class Calculator:
    """Tool for performing mathematical calculations"""
    
    def __init__(self):
        # Define supported operations and their corresponding functions
        self.operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else "Error: Division by zero",
            '^': lambda x, y: x ** y,
            'sqrt': lambda x, _: math.sqrt(x) if x >= 0 else "Error: Cannot calculate square root of negative number",
            'sin': lambda x, _: math.sin(math.radians(x)),
            'cos': lambda x, _: math.cos(math.radians(x)),
            'tan': lambda x, _: math.tan(math.radians(x)),
        }
    
    def execute(self, query: str) -> str:
        """
        Extract and solve a mathematical expression from the query
        
        Args:
            query: The user query containing a mathematical expression
            
        Returns:
            String with the calculation result
        """
        # First, try to extract a basic arithmetic expression
        basic_expr = re.search(r'\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?', query)
        if basic_expr:
            expression = basic_expr.group()
            return self._evaluate_basic_expression(expression)
        
        # Try to extract function calls like sqrt, sin, cos, etc.
        func_expr = re.search(r'(sqrt|sin|cos|tan)\s*\(\s*\d+(?:\.\d+)?\s*\)', query)
        if func_expr:
            expression = func_expr.group()
            return self._evaluate_function(expression)
        
        return "Could not identify a valid mathematical expression in the query."
    
    def _evaluate_basic_expression(self, expression: str) -> str:
        """Evaluate a basic arithmetic expression"""
        try:
            # Extract numbers and operator
            match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/\^])\s*(\d+(?:\.\d+)?)', expression)
            if not match:
                return f"Could not parse expression: {expression}"
            
            num1 = float(match.group(1))
            operator = match.group(2)
            num2 = float(match.group(3))
            
            # Check if operator is supported
            if operator not in self.operations:
                return f"Unsupported operator: {operator}"
            
            # Calculate result
            result = self.operations[operator](num1, num2)
            
            # Format result
            if isinstance(result, float) and result.is_integer():
                result = int(result)
                
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating result: {str(e)}"
    
    def _evaluate_function(self, expression: str) -> str:
        """Evaluate a function call like sqrt(x)"""
        try:
            # Extract function name and argument
            match = re.search(r'(sqrt|sin|cos|tan)\s*\(\s*(\d+(?:\.\d+)?)\s*\)', expression)
            if not match:
                return f"Could not parse function: {expression}"
            
            func_name = match.group(1)
            argument = float(match.group(2))
            
            # Check if function is supported
            if func_name not in self.operations:
                return f"Unsupported function: {func_name}"
            
            # Calculate result (using None as second parameter for unary functions)
            result = self.operations[func_name](argument, None)
            
            # Format result
            if isinstance(result, float):
                result = round(result, 6)
                if result.is_integer():
                    result = int(result)
                    
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating result: {str(e)}"