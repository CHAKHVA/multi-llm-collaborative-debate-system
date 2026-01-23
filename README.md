# Multi-LLM Collaborative Debate System

A multi-agent system where LLM agents collaborate through structured debate to solve complex reasoning problems. Each problem goes through 5 stages: role assignment, independent solutions, peer review, refinement, and final judgment.

## Quick Start

**Prerequisites**: Python 3.10+, OpenAI API key

1. Clone and setup:

```bash
git clone https://github.com/CHAKHVA/multi-llm-collaborative-debate-system.git
cd multi-llm-collaborative-debate-system
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Configure API key:

```bash
echo 'OPENAI_API_KEY="your-api-key-here"' > .env
```

## How to Run

```bash
# Run full experiment (25 problems)
python main.py

# Run single problem
python main.py --test-id 1

# Interactive demo
jupyter notebook notebooks/demo_playground.ipynb

# View results analysis
jupyter notebook notebooks/analysis.ipynb
```

## How It Works

- **3 Solver agents** independently solve problems, then critique each other
- **1 Judge agent** evaluates solutions and selects the best answer
- Each agent has a **distinct persona** (Scientist, Strategist, Engineer, Mediator) to encourage diverse reasoning
- Solutions are **refined** based on peer feedback before final judgment

Results are saved to `data/results_log.json`.
