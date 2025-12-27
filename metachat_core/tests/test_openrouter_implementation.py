import os
from unittest.mock import MagicMock, patch
import asyncio
import sys

# Mock dependencies before import
sys.modules["tiktoken"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["together"] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

async def test_openrouter_model_instantiation():
    try:
        from metachat_core.core.models.openrouter import OpenRouterModel
    except ImportError as e:
        raise ImportError(f"Could not import OpenRouterModel: {e}")

    model = OpenRouterModel(model_name="test-model", api_key="test-key")
    assert model.model_name == "test-model"
    assert hasattr(model, "generate")
    assert hasattr(model, "count_tokens")

async def test_openrouter_model_generate():
    from metachat_core.core.models.openrouter import OpenRouterModel
        
    model = OpenRouterModel(model_name="test-model", api_key="test-key")
    
    # Mock the internal client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenRouter"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    
    # Mock the client
    # We need to know the structure of OpenRouterModel to mock correctly.
    # Assuming it has self.client like OpenAIModel
    
    # Create a Future for the return value to support await
    f = asyncio.Future()
    f.set_result(mock_response)
    
    with patch.object(model.client.chat.completions, 'create', return_value=f) as mock_create:
        messages = [{"role": "user", "content": "Hello"}]
        response = await model.generate(messages)
        
        assert response.content == "Hello from OpenRouter"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

if __name__ == "__main__":
    try:
        asyncio.run(test_openrouter_model_instantiation())
        asyncio.run(test_openrouter_model_generate())
        print("Tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
