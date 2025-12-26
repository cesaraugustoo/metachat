import os
import sys

def test_pyproject_toml_exists():
    if not os.path.exists("metachat_core/pyproject.toml"):
        print("AssertionError: metachat_core/pyproject.toml does not exist")
        sys.exit(1)
    print("Test passed")

if __name__ == "__main__":
    test_pyproject_toml_exists()