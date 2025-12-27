import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before import
sys.modules["tiktoken"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["together"] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_export():
    try:
        from metachat_core.core.models import OpenRouterModel
        print("Import successful!")
    except ImportError:
        print("Import failed!")
        sys.exit(1)

if __name__ == "__main__":
    test_export()
