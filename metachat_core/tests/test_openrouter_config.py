import sys
import os
from unittest.mock import MagicMock

# Mock pydantic if not available
try:
    from pydantic_settings import BaseSettings as PydanticBaseSettings
except ImportError:
    class PydanticBaseSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    sys.modules["pydantic_settings"] = MagicMock()
    sys.modules["pydantic_settings"].BaseSettings = PydanticBaseSettings

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(default=..., **kwargs):
        return default
    sys.modules["pydantic"] = MagicMock()
    sys.modules["pydantic"].BaseModel = BaseModel
    sys.modules["pydantic"].Field = Field

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_openrouter_api_settings():
    from metachat_core.config.settings import APISettings
    
    apis = APISettings()
    
    # These should fail if the fields are not defined in the class
    print(f"Checking for openrouter_api_key...")
    assert hasattr(apis, "openrouter_api_key"), "openrouter_api_key missing from APISettings"
    print(f"Checking for openrouter_model_name...")
    assert hasattr(apis, "openrouter_model_name"), "openrouter_model_name missing from APISettings"
    
    assert apis.openrouter_api_key is None
    assert apis.openrouter_model_name == "google/gemini-2.0-flash-001"

if __name__ == "__main__":
    try:
        test_openrouter_api_settings()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
