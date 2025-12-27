import os
import sys
import asyncio
from unittest.mock import MagicMock, patch

# Mock dependencies before import
sys.modules["tiktoken"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["together"] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from metachat_core.core.models.openrouter import OpenRouterModel

async def test_openrouter_model_instantiation():
    print("Testing instantiation...")
    model = OpenRouterModel(model_name="test-model", api_key="test-key")
    assert model.model_name == "test-model"
    assert hasattr(model, "generate")
    assert hasattr(model, "count_tokens")

async def test_openrouter_model_generate():
    print("Testing generate...")
    model = OpenRouterModel(model_name="test-model", api_key="test-key")
    
    # Mock the internal client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenRouter"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    
    # Create a Future for the return value to support await
    f = asyncio.Future()
    f.set_result(mock_response)
    
    with patch.object(model.client.chat.completions, 'create', return_value=f) as mock_create:
        messages = [{"role": "user", "content": "Hello"}]
        response = await model.generate(messages)
        
        assert response.content == "Hello from OpenRouter"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        
    from openai import AsyncOpenAI
    
    # Verify AsyncOpenAI was called with correct base_url
    # AsyncOpenAI is a MagicMock from our sys.modules mock
    AsyncOpenAI.assert_called_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1"
    )

async def test_openrouter_count_tokens():
    print("Testing count_tokens...")
    model = OpenRouterModel(model_name="test-model", api_key="test-key")
    
    import tiktoken
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3] # 3 tokens
    
    with patch("tiktoken.get_encoding", return_value=mock_encoding):
        count = model.count_tokens("Hello world")
        assert count == 3
        tiktoken.get_encoding.assert_called_with("cl100k_base")

if __name__ == "__main__":
    try:
        asyncio.run(test_openrouter_model_instantiation())
        asyncio.run(test_openrouter_model_generate())
        asyncio.run(test_openrouter_count_tokens())
        print("All OpenRouter unit tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
