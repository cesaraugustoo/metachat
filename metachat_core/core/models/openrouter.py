from typing import List, Dict, Any, Optional
import os
import tiktoken
from openai import AsyncOpenAI
from .base import BaseModel, LLMResponse

class OpenRouterModel(BaseModel):
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        # Default to environment variable if not provided, else fallback to a default model
        model_name = model_name or os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-2.0-flash-001")
        super().__init__(model_name)
        
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            # We don't raise an error here to allow for instantiation without key if user wants to set it later or mocking,
            # but in practice it's needed.
            pass

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    
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
        
        # Include optional OpenRouter headers if configured (though typically handled by client config if wrapper used)
        # Here we just use the standard OpenAI client which doesn't automatically add them.
        # For full compliance, one might add extra_headers, but the spec didn't mandate them strictly beyond what we discussed.
        # We'll stick to basic implementation.

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
        # OpenRouter hosts many models. tiktoken cl100k_base is a reasonable default for modern ones.
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if cl100k_base fails for some reason
            encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
