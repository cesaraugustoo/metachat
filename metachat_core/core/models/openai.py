from typing import List, Dict, Any, Optional
import os
import tiktoken
from openai import AsyncOpenAI
from .base import BaseModel, LLMResponse

class OpenAIModel(BaseModel):
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-5.2-2025-12-11")
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    async def generate(self, 
                      messages: List[Dict[str, Any]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> LLMResponse:
        
        processed_messages = []
        for message in messages:
            if 'images' in message:
                content = []
                if 'content' in message and message['content']:
                    content.append({"type": "text", "text": message['content']})
                for image_data in message['images']:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    })
                processed_messages.append({"role": message["role"], "content": content})
            else:
                # Ensure we only pass role and content to the API
                processed_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=processed_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            raw_response=response,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
    
    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback for models not yet in tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
