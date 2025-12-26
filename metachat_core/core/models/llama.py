from typing import List, Dict, Any, Optional
from together import Together
from .base import BaseModel, LLMResponse

class LlamaModel(BaseModel):
    """Llama API model implementation via Together AI"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name)
        # Together() will look for TOGETHER_API_KEY env var if api_key not provided
        self.client = Together(api_key=api_key)
    
    async def generate(self, 
                      messages: List[Dict[str, Any]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate a response using the Together API"""
        
        # Create API request
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        return LLMResponse(
            content=content,
            raw_response=response,
            input_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
            output_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else None
        )
    
    def count_tokens(self, text: str) -> int:
        # Rough token estimate
        return int(len(text.split()) * 1.3)
