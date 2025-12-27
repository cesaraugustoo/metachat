# metachat-aim

Code for reproducing the evaluation of AIM MetaChat on the Stanford Nanophotonics Benchmark from [*A multi-agentic framework for real-time, autonomous freeform metasurface design*](https://www.science.org/doi/10.1126/sciadv.adx8006).

![AIM evaluation overview](../figs/fig2.png)

## Setup

Please refer to the [Centralized User Guide](../USER_GUIDE.md) for general installation and configuration (API keys, environments).

## Directory Structure

- `agent/`: Agent implementations (e.g., `StandardAgent`, *AIM* `IterativeAgent`).
- `core/models/`: LLM wrappers (`OpenAIModel`, `AnthropicModel`, `LlamaModel`).
- `tools/`: Domain tools (design utilities, materials DB, solvers).
- `experiments/`: Benchmarks, evaluation framework, and runner scripts.

## Running Benchmark Evaluations

1. Ensure your environment is set up according to the User Guide.
2. Run the benchmark evaluation:
   ```bash
   python experiments/runners/eval_runner.py
   ```

To change the agent or models, edit the dictionary in `experiments/runners/eval_runner.py`.