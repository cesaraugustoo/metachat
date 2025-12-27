# Plan: vLLM Support for Local Models

## Phase 1: Configuration Updates [checkpoint: fbfc704]
- [x] Task: Update `APISettings` in `metachat_core/config/settings.py` to include `vllm_base_url` and `vllm_model_name`. (fbfc704)
- [x] Task: Update `.env.example` with vLLM configuration templates. (fbfc704)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration Updates' (Protocol in workflow.md) (fbfc704)

## Phase 2: vLLM Adapter Development (TDD) [checkpoint: 46c6f47]
- [x] Task: Create `metachat_core/tests/test_vllm_model.py` and write failing tests for `VLLMModel` initialization and health check. (46c6f47)
- [x] Task: Implement `VLLMModel.__init__` with health check in `metachat_core/core/models/vllm.py`. (46c6f47)
- [x] Task: Write failing tests for `VLLMModel.generate` (supporting stop sequences and error handling). (46c6f47)
- [x] Task: Implement `VLLMModel.generate` using the OpenAI client. (46c6f47)
- [x] Task: Write failing tests for `VLLMModel.count_tokens`. (46c6f47)
- [x] Task: Implement `VLLMModel.count_tokens` with fallback logic. (46c6f47)
- [x] Task: Conductor - User Manual Verification 'Phase 2: vLLM Adapter Development (TDD)' (Protocol in workflow.md) (46c6f47)

## Phase 3: Integration and Finalization
- [ ] Task: Export `VLLMModel` in `metachat_core/core/models/__init__.py`.
- [ ] Task: Verify overall `metachat_core` test suite and coverage.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Finalization' (Protocol in workflow.md)
