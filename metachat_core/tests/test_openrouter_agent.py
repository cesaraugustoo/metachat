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
from metachat_core.agent.cot_iterative import IterativeAgent

async def test_openrouter_agent_integration():
    print("Testing OpenRouter agent integration...")
    model = OpenRouterModel(model_name="test-model", api_key="test-key")
    agent = IterativeAgent(model=model)
    
    # Mock the internal client response
    mock_raw_response = MagicMock()
    mock_raw_response.choices = [MagicMock()]
    mock_raw_response.choices[0].message.content = "I have thought about it. <response>The answer is 42</response>"
    mock_raw_response.usage.prompt_tokens = 10
    mock_raw_response.usage.completion_tokens = 5
    
    f = asyncio.Future()
    f.set_result(mock_raw_response)
    
    with patch.object(model.client.chat.completions, 'create', return_value=f):
        result = await agent.solve("What is the meaning of life?")
        print(f"Agent result: {result}")
        
        assert "solution" in result, f"Result missing solution: {result}"
        assert result["solution"] == "The answer is 42"
        assert result["metadata"]["num_iterations"] == 1
        print("Integration test passed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_openrouter_agent_integration())
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
