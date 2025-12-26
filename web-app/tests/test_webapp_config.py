import sys
import os
from unittest.mock import MagicMock, patch

# Mock all external dependencies
for mod in ["openai", "anthropic", "together", "tiktoken", "numpy", "scipy", "sympy", "gdspy", "pandas", "aiofiles", "tqdm", "sse_starlette", "fastapi", "python-dotenv", "dotenv", "pydantic", "pydantic_settings"]:
    mock = MagicMock()
    sys.modules[mod] = mock

# Manually mock Pydantic base classes
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

class MockBaseSettings(MockBaseModel):
    pass

sys.modules["pydantic"].BaseModel = MockBaseModel
sys.modules["pydantic"].Field = MagicMock(return_value=None)
sys.modules["pydantic_settings"].BaseSettings = MockBaseSettings

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def test_webapp_config_integration():
    print("Testing Web-App config integration...")
    try:
        from metachat_core.config import get_settings
        assert get_settings is not None
        print("Import check: OK")
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_webapp_config_integration()
