# Inter-component Communication and Data Flow

## Overview
MetaChat's data flow is characterized by an interactive "monologue" loop where an LLM-based agent orchestrates high-level reasoning and low-level scientific tool execution.

## 1. Interaction Loop (AIM)
1. **Request:** The user provides a high-level design goal (e.g., "Design a metalens for 800nm").
2. **Reasoning:** The `IterativeAgent` processes the request, potentially talking to itself in multiple iterations to plan the design parameters.
3. **Tool Call:** If a design is required, the agent generates a special XML-like tag: `<tool>neural_design ... </tool>`.
4. **Parsing:** The agent's `solve()` loop detects the tag and routes the string to the `NeuralDesignAPI.execute()` method.

## 2. Tool Orchestration
### Neural Design API (`tools/design/api.py`)
- **Parsing:** Uses Python's `ast` module to safely parse the arguments from the agent's code block.
- **Validation:** Performs type checking on parameters (wavelength, refractive index, etc.).
- **Execution (Two Flavors):**
    - **Standalone Agent (`metachat-aim/`):** Returns a confirmation string indicating the API was called.
    - **Web App Backend (`web-app/`):**
        1. **Script Generation:** Generates a temporary Python script that imports `superpixel_optimization_gpu_pared`.
        2. **Containerization:** Executes this script inside a Docker container (`rclupoiu/waveynet:metachat`) to ensure a consistent environment with GPU access.
        3. **Result Collection:** Monitors a `progress.txt` file and collects output plots (`.png`) and device patterns (`.npy`).
        4. **GDS Conversion:** Converts the `.npy` pattern to a GDSII file using `gdspy`.
        5. **Persistence:** Saves design metadata and file paths to a SQLite `DesignDatabase`.
        6. **Response:** Returns base64-encoded plots and a download URL for the GDS file back to the agent.

## 3. Data Flow Diagrams

### High-Level Flow
`User -> Web App UI -> FastAPI -> IterativeAgent -> LLM`
`LLM -> IterativeAgent -> NeuralDesignAPI -> Docker (GPU) -> Solver`
`Solver -> Results Dir -> NeuralDesignAPI -> GDS File + Plots`
`GDS + Plots -> IterativeAgent -> FastAPI (SSE) -> Web App UI -> User`

### Solver Communication (Web App)
- **FastAPI Backend** acts as a controller.
- **Docker** provides the execution sandbox for heavy scientific compute.
- **Shared Volumes (`/media`)** allow the host and container to exchange large files (checkpoints, results).
- **SQLite** maintains a persistent registry of all generated designs.
