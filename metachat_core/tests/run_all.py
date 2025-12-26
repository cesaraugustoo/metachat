import sys
import os
from unittest.mock import MagicMock

# Mock all external dependencies
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["together"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["gdspy"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from metachat_core.core.models import BaseModel, OpenAIModel
from metachat_core.agent import Agent, IterativeAgent
from metachat_core.core.tools import BaseTool, ToolCall
from metachat_core.core.tools.design import NeuralDesignAPI

def run_tests():
    print("Running Component Migration Verification...")
    
    # 1. Models
    assert issubclass(OpenAIModel, BaseModel)
    print("Models verified.")

    # 2. Tools
    assert issubclass(NeuralDesignAPI, BaseTool)
    print("Tools verified.")

    # 3. Agents
    mock_model = MagicMock(spec=BaseModel)
    mock_model.model_name = "test"
    agent = IterativeAgent(model=mock_model)
    assert isinstance(agent, Agent)
    print("Agents verified.")

    print("All component structures verified successfully!")

if __name__ == "__main__":
    run_tests()
