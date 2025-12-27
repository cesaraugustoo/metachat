# Specification: OpenRouter Support

## Overview
This track adds support for OpenRouter as a model provider in `metachat_core`. This allows the framework to leverage a vast array of models available through the OpenRouter API while maintaining compatibility with the existing agentic workflows.

## Functional Requirements
1.  **New Model Class:** Implement `OpenRouterModel` in `metachat_core/core/models/openrouter.py`.
    -   Inherit from `BaseModel`.
    -   Use `AsyncOpenAI` client (OpenRouter is OpenAI-compatible).
    -   Base URL must be set to `https://openrouter.ai/api/v1`.
2.  **Configuration:** Update `APISettings` in `metachat_core/config/settings.py` to include:
    -   `openrouter_api_key`: Optional string, default `None`.
    -   `openrouter_model_name`: String, default (e.g., `google/gemini-2.0-flash-001`).
3.  **Exporting:** Add `OpenRouterModel` to `metachat_core/core/models/__init__.py`.
4.  **Token Counting:** Implement `count_tokens` using `tiktoken` with `cl100k_base` as the default encoding for OpenRouter models.

## Non-Functional Requirements
- **Maintainability:** Ensure the implementation follows the pattern established by `OpenAIModel`.
- **Error Handling:** Gracefully handle API errors and missing configuration.

## Acceptance Criteria
- `OpenRouterModel` can be instantiated with settings from environment variables (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL_NAME`).
- `OpenRouterModel.generate()` successfully returns an `LLMResponse`.
- `OpenRouterModel.count_tokens()` returns an accurate (or best-effort) token count.
- The new model is compatible with `IterativeAgent`.

## Out of Scope
- Implementing OpenRouter-specific features like "rankings" or "credits" API.
- Updating the web-app UI to switch between providers (this will be handled by environment configuration for now).
