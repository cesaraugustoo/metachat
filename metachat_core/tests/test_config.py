import sys
import os
from unittest.mock import MagicMock

# Mock pydantic dependencies for environment without them installed
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

def test_config_implementation():
    # Attempt import
    from metachat_core.config.settings import PathSettings, APISettings, SolverSettings, BaseSettings
    
    print("Testing PathSettings...")
    paths = PathSettings(data_dir="/tmp/data", log_dir="/tmp/logs")
    assert paths.data_dir == "/tmp/data"
    
    print("Testing APISettings...")
    apis = APISettings(openai_api_key="sk-test")
    assert apis.openai_api_key == "sk-test"
    
    print("Testing SolverSettings...")
    solver = SolverSettings(batch_size=64)
    assert solver.batch_size == 64
    
    print("Testing BaseSettings composition...")
    # BaseSettings should optionally take sub-settings or load them
    settings = BaseSettings(
        paths=paths,
        api=apis,
        solver=solver
    )
    assert settings.paths.data_dir == "/tmp/data"
    assert settings.solver.batch_size == 64
    
    print("Config tests passed!")

if __name__ == "__main__":
    test_config_implementation()
