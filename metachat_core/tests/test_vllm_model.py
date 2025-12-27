import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Mock dependencies before any metachat_core imports
sys.modules["anthropic"] = MagicMock()
sys.modules["together"] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestVLLMModel(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_name = "test-vllm-model"
        self.base_url = "http://localhost:8000/v1"
        
        # Patch AsyncOpenAI
        self.openai_patcher = patch('metachat_core.core.models.vllm.AsyncOpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Patch httpx
        self.httpx_patcher = patch('metachat_core.core.models.vllm.httpx')
        self.mock_httpx = self.httpx_patcher.start()
        # Default mock response for httpx
        self.mock_response = MagicMock()
        self.mock_httpx.get.return_value = self.mock_response
        
        # Patch get_settings
        self.settings_patcher = patch('metachat_core.core.models.vllm.get_settings')
        self.mock_get_settings = self.settings_patcher.start()
        self.mock_settings = MagicMock()
        self.mock_settings.api.vllm_base_url = self.base_url
        self.mock_settings.api.vllm_model_name = self.model_name
        self.mock_get_settings.return_value = self.mock_settings

    def tearDown(self):
        self.openai_patcher.stop()
        self.httpx_patcher.stop()
        self.settings_patcher.stop()

    async def test_initialization_success(self):
        from metachat_core.core.models.vllm import VLLMModel
        
        # Mock models.list() for health check (not used in this test because loop is running)
        self.mock_openai.return_value.models.list = AsyncMock()
        
        model = VLLMModel()
        self.assertEqual(model.model_name, self.model_name)
        self.mock_openai.assert_called_once_with(
            base_url=self.base_url,
            api_key="none"
        )
        # Verify httpx was used because loop is running
        self.mock_httpx.get.assert_called_once_with(f"{self.base_url}/models")
        self.mock_response.raise_for_status.assert_called_once()

    async def test_initialization_failure(self):
        from metachat_core.core.models.vllm import VLLMModel
        
        # Mock httpx.get to raise exception
        self.mock_httpx.get.side_effect = Exception("Connection refused")
        
        with self.assertRaises(ConnectionError) as cm:
            VLLMModel()
        self.assertIn("Could not connect to vLLM server", str(cm.exception))

    async def test_generate_success(self):
        from metachat_core.core.models.vllm import VLLMModel
        
        model = VLLMModel()
        
        # Mock chat.completions.create
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        
        model.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "test message"}]
        response = await model.generate(messages, temperature=0.5, max_tokens=100)
        
        self.assertEqual(response.content, "Generated text")
        self.assertEqual(response.input_tokens, 10)
        self.assertEqual(response.output_tokens, 20)
        
        model.client.chat.completions.create.assert_awaited_once_with(
            model=self.model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=100,
            stop=None
        )

    async def test_generate_with_stop_sequences(self):
        from metachat_core.core.models.vllm import VLLMModel
        
        model = VLLMModel()
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Stopped"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 2
        
        model.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "test"}]
        stop_sequences = ["\n", "User:"]
        await model.generate(messages, stop=stop_sequences)
        
        model.client.chat.completions.create.assert_awaited_once_with(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=None,
            stop=stop_sequences
        )

    async def test_generate_failure(self):
        from metachat_core.core.models.vllm import VLLMModel
        
        model = VLLMModel()
        
        model.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        messages = [{"role": "user", "content": "test"}]
        with self.assertRaises(Exception) as cm:
            await model.generate(messages)
        self.assertIn("API Error", str(cm.exception))

    async def test_count_tokens(self):
        from metachat_core.core.models.vllm import VLLMModel
        
        model = VLLMModel()
        text = "Hello, world!"
        token_count = model.count_tokens(text)
        
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)

if __name__ == '__main__':
    unittest.main()