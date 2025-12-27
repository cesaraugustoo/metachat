# Specification: vLLM Support for Local Models

## Overview
This track adds support for local Large Language Models (LLMs) via the `vLLM` inference engine. Integration will be achieved by creating a dedicated adapter that interacts with vLLM's OpenAI-compatible API server. This allows users to leverage high-performance local inference while maintaining a consistent interface with existing cloud-based providers.

## Functional Requirements
- **Dedicated Adapter:** Implement `VLLMModel` in `metachat_core/core/models/vllm.py`, inheriting from `BaseModel`.
- **OpenAI Compatibility:** Utilize the `openai` Python client to communicate with the vLLM server.
- **Configurable Endpoint:** Support configurable `vllm_base_url` and `vllm_model_name` via `APISettings`.
- **Connection Validation:** Perform a basic health check (e.g., a simple version or models list request) during initialization to ensure the vLLM server is reachable.
- **Custom Stop Sequences:** Ensure the `generate` method correctly passes stop sequences to the vLLM API.
- **Token Counting:** Implement token counting, potentially using a generic encoder (like `cl100k_base`) as a fallback if the specific model tokenizer is not available locally.

## Non-Functional Requirements
- **Consistency:** Mimic the structure and error handling of the existing `OpenAIModel` and `AnthropicModel`.
- **Testability:** Include unit tests using mocks for the vLLM API responses.
- **Documentation:** Update relevant configuration documentation to include vLLM setup instructions.

## Acceptance Criteria
- [ ] `VLLMModel` is implemented and can successfully send requests to a mock vLLM server.
- [ ] `APISettings` includes `vllm_base_url` and `vllm_model_name`.
- [ ] The system correctly handles connection errors if the vLLM server is offline.
- [ ] Unit tests cover successful generation and connection failure scenarios.
- [ ] `metachat_core/core/models/__init__.py` is updated to export the new model.

## Out of Scope
- Support for vLLM-specific sampling parameters beyond standard OpenAI-compatible ones (except stop sequences).
- Automated lifecycle management (starting/stopping) of the vLLM server itself.
- Support for local model quantization settings within the adapter (managed at the vLLM server level).
