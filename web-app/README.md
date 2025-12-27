# MetaChat Web App

This MetaChat web app example code demonstrates how to run an AIM design and materials agent on a backend that is interfaced via a simple frontend chat window.

## Setup and Usage

Detailed instructions for setting up the environment, installing dependencies, and running the application can be found in the [Centralized User Guide](../USER_GUIDE.md).

## Extending the Web App

These examples can be extended to different devices by following these steps:
1) Adding more APIs in `backend/tools/design/superpixel_optimization_gpu_pared.py`
2) Exposing them to the AIM agent in `backend/tools/design/api.py`
3) Adding them to the AIM agent prompt in `backend/agent/cot_iterative_tools_materials.py`

## Project Structure

- `frontend/`: Simple HTML/CSS/JavaScript interface.
- `backend/`: FastAPI server handling chat requests and agent orchestration.
- `main.py`: Entry point for the backend server.