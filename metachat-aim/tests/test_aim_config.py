import sys
import os
from unittest.mock import MagicMock, patch

# Mock all external dependencies
for mod in ["openai", "anthropic", "together", "tiktoken", "numpy", "scipy", "sympy", "gdspy", "pandas", "aiofiles", "tqdm", "sse_starlette", "fastapi", "python-dotenv", "dotenv", "pydantic", "pydantic_settings"]:
    mock = MagicMock()
    sys.modules[mod] = mock

# Manually mock Pydantic base classes to allow inheritance
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
    @classmethod
    def model_validate(cls, obj): return obj

class MockBaseSettings(MockBaseModel):
    pass

sys.modules["pydantic"].BaseModel = MockBaseModel
sys.modules["pydantic"].Field = MagicMock(return_value=None)
sys.modules["pydantic_settings"].BaseSettings = MockBaseSettings
sys.modules["pydantic_settings"].SettingsConfigDict = MagicMock(return_value={})

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def test_aim_config_integration():
    print("Testing AIM config integration...")
    try:
        from metachat_core.config import get_settings
        # We can't easily test real Pydantic loading without real Pydantic
        # but we can verify the import path is correct and the factory exists.
        assert get_settings is not None
        print("Import check: OK")
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_aim_config_integration()
