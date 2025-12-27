import os
import sys
import unittest
from metachat_core.config.settings import APISettings

class TestVLLMConfig(unittest.TestCase):
    def test_vllm_settings_defaults(self):
        """Test default values for vLLM settings"""
        settings = APISettings()
        # These should fail initially as fields don't exist
        self.assertTrue(hasattr(settings, 'vllm_base_url'))
        self.assertTrue(hasattr(settings, 'vllm_model_name'))
        self.assertEqual(settings.vllm_base_url, "http://localhost:8000/v1")
        self.assertEqual(settings.vllm_model_name, "meta-llama/Llama-2-7b-chat-hf")

    def test_vllm_settings_custom(self):
        """Test custom values for vLLM settings"""
        settings = APISettings(
            vllm_base_url="http://192.168.1.100:8000/v1",
            vllm_model_name="mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.assertEqual(settings.vllm_base_url, "http://192.168.1.100:8000/v1")
        self.assertEqual(settings.vllm_model_name, "mistralai/Mistral-7B-Instruct-v0.2")

if __name__ == '__main__':
    unittest.main()
