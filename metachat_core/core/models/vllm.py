from typing import List, Dict, Any, Optional
import asyncio
import tiktoken
import httpx
from openai import AsyncOpenAI
from .base import BaseModel, LLMResponse
from ...config.factory import get_settings

class VLLMModel(BaseModel):
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        settings = get_settings()
        model_name = model_name or settings.api.vllm_model_name
        self.base_url = base_url or settings.api.vllm_base_url
        
        super().__init__(model_name)
        
        # vLLM usually doesn't require an API key, but AsyncOpenAI requires one
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="none"
        )
        
        # Health check
        try:
            # We use a simple models.list() as a health check.
            # Since __init__ is sync, we need to run this briefly.
            # In a production async app, this might be better as an async setup method,
            # but per spec we do it in init.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                response = httpx.get(f"{self.base_url}/models")
                response.raise_for_status()
            else:
                asyncio.run(self.client.models.list())
        except Exception as e:
            raise ConnectionError(f"Could not connect to vLLM server at {self.base_url}: {e}")

    async def generate(self, 
                      messages: List[Dict[str, Any]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None,
                      stop: Optional[List[str]] = None) -> LLMResponse:
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            raw_response=response,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
    
    def count_tokens(self, text: str) -> int:
        try:
            # Try to get encoding for the model
            encoding = tiktoken.encoding_for_model(self.model_name)
        except (KeyError, ValueError):
            # Fallback for local models not in tiktoken database
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
