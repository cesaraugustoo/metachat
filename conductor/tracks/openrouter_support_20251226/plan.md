# Plan: OpenRouter Support

## Phase 1: Configuration
- [x] Task: Update `APISettings` in `metachat_core/config/settings.py` to include `openrouter_api_key` and `openrouter_model_name`. c608f10
- [x] Task: Update `.env.example` with `OPENROUTER_API_KEY` and `OPENROUTER_MODEL_NAME`. 3b4bb6d
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration' (Protocol in workflow.md) [checkpoint: a2f3dad]

## Phase 2: Core Implementation
- [x] Task: Implement `OpenRouterModel` in `metachat_core/core/models/openrouter.py` inheriting from `BaseModel`. 8dc087a
- [x] Task: Export `OpenRouterModel` in `metachat_core/core/models/__init__.py`. 8dc087a
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core Implementation' (Protocol in workflow.md)

## Phase 3: Testing and Verification
- [ ] Task: Write unit tests for `OpenRouterModel` in `metachat_core/tests/test_openrouter.py`.
- [ ] Task: Verify `OpenRouterModel` compatibility with `IterativeAgent` in an integration test.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Testing and Verification' (Protocol in workflow.md)
