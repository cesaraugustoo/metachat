# MetaChat User Guide

Welcome to the centralized guide for setting up and using MetaChat, a multi-agentic framework for real-time, autonomous freeform metasurface design.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration](#configuration)
4. [Downloading Data and Models](#downloading-data-and-models)
5. [Running the Application](#running-the-application)
6. [Usage Tutorial](#usage-tutorial)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure your system meets the following requirements:
- **Operating System:** Linux (recommended) or macOS.
- **Hardware:** At least one NVIDIA GPU with matching CUDA drivers (required for solvers).
- **Python:** Version 3.10 or higher.
- **Tools:**
  - [Poetry](https://python-poetry.org/docs/#installation) (Python package manager)
  - [Docker](https://docs.docker.com/get-docker/) (for WaveY-Net solvers)
  - [Git](https://git-scm.com/downloads)

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jonfanlab/metachat.git
cd metachat
```

### 2. Install Core Dependencies
MetaChat is organized as a monorepo. You should install dependencies for the core library and the web application.

**Core Library:**
```bash
cd metachat_core
poetry install
cd ..
```

**Web Application:**
```bash
cd web-app
poetry install
cd ..
```

**AIM Evaluation Stack (Optional):**
```bash
cd metachat-aim
pip install -r requirements.txt
cd ..
```

### 3. Pull WaveY-Net Docker Image
The backend runs GPU simulation jobs inside a Docker container.
```bash
docker pull rclupoiu/waveynet:metachat
```

---

## Configuration

### Environment Variables
MetaChat uses a centralized configuration system. Copy the template to create your `.env` file at the root:
```bash
cp .env.example .env
```

Edit the `.env` file and provide your API keys and path overrides:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `APP_ENV`: Set to `dev`, `prod`, or `docker`.
- `GPU_IDS`: Comma-separated list of GPU IDs (e.g., `"0,1"`).

For a full list of configuration options, see [docs/config.md](docs/config.md).

---

## Downloading Data and Models

### 1. WaveY-Net Checkpoints
Download the pretrained weights and scaling factors from Zenodo:
```bash
mkdir -p data/waveynet
cd data/waveynet
curl -L -o metachat_code_data.zip "https://zenodo.org/records/15802727/files/metachat_code_data.zip?download=1"
unzip -q metachat_code_data.zip
```
Then set `CHECKPOINT_DIRECTORY_MULTISRC` in your `.env` to the absolute path of this directory.

### 2. Materials Database
Download the `materials.db` file from [Zenodo](https://zenodo.org/records/15802727) and place it in the path specified by `MATERIAL_DB_PATH` in your `.env`.

---

## Running the Application

### 1. Start the Backend
```bash
cd web-app
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
In a new terminal:
```bash
cd web-app/frontend
python3 -m http.server 8080
```

### 3. Access the UI
Open your browser and navigate to `http://localhost:8080`.

---

## Usage Tutorial

### Designing a Metasurface
1. **Enter Requirements:** In the chat interface, describe your design goals (e.g., "Design a silicon-on-insulator deflector for 1550nm at 40 degrees").
2. **Agent Collaboration:** Observe the AIM agents (Design Agent, Materials Agent) collaborating to propose structures.
3. **Simulation:** The agents will automatically trigger FiLM WaveY-Net simulations to verify performance.
4. **Iterative Refinement:** The agents will refine the freeform structure based on simulation results until goals are met.

---

## Troubleshooting

### Port Already in Use
If port 8000 or 8080 is occupied:
```bash
sudo lsof -i :8000
kill -9 <PID>
```

### Docker Permissions
Ensure your user is in the `docker` group:
```bash
sudo usermod -aG docker $USER
```
(Log out and back in for changes to take effect).
