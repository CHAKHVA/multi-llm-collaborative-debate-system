# Multi-LLM Collaborative Debate System

A Multi-Agent System (MAS) where multiple LLM agents collaborate through structured debate to solve complex reasoning problems. The system leverages peer review and iterative refinement to improve answer quality.

## Overview

This project implements a collaborative debate framework where:

- **3 Solver agents** independently solve problems, then critique each other's work
- **1 Judge agent** evaluates all solutions and selects the best answer
- All agents use **distinct personas** to prevent mode collapse
- Solutions are **refined** based on peer feedback before final judgment

### Key Features

- **Structured Debate Protocol**: 5-stage workflow (Role Assignment → Solution → Peer Review → Refinement → Verdict)
- **Diverse Agent Personas**: Four distinct thinking styles to encourage varied approaches
- **Peer Review System**: Round-robin critiques with scoring and error identification
- **LLM-as-a-Judge Evaluation**: Automated grading against ground truth
- **Robust API Handling**: Retry logic with exponential backoff via `tenacity`

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE 0: Role Assignment                 │
│  All 4 agents express preference (Solver/Judge) + confidence    │
│  → 1 Judge selected, 3 Solvers assigned                         │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Independent Solutions               │
│  Each Solver generates a solution independently (no comms)      │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Peer Review (Round Robin)           │
│  Solver A → reviews B, C                                        │
│  Solver B → reviews A, C                                        │
│  Solver C → reviews A, B     (6 reviews total)                  │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                       STAGE 3: Refinement                       │
│  Each Solver improves their solution based on received feedback │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                      STAGE 4: Final Verdict                     │
│  Judge evaluates all solutions + critiques → selects winner     │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Personas

To prevent "mode collapse" (where all agents converge to identical reasoning), each agent has a distinct persona:

| Agent | Persona | Thinking Style |
| --- | --- | --- |
| A | The Rigorous Scientist | Formal proofs, step-by-step derivation, skeptical of intuition |
| B | The Creative Strategist | Lateral thinking, alternative interpretations, outside-the-box |
| C | The Practical Engineer | Probability, heuristics, real-world constraints, intuition checks |
| D | The Generalist Mediator | Balanced synthesis, clarity, bridges logic and application |

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/CHAKHVA/multi-llm-collaborative-debate-system.git
cd multi-llm-collaborative-debate-system
```

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure your OpenAI API key:

```bash
# Create a .env file
echo 'OPENAI_API_KEY="your-api-key-here"' > .env
```

## Usage

### Run Full Experiment

Process all 25 problems in the dataset:

```bash
python main.py
```

### Run Single Problem

Test on a specific problem by ID:

```bash
python main.py --test-id 1
```

### Interactive Demo (Jupyter)

For step-by-step exploration:

```bash
jupyter notebook notebooks/demo_playground.ipynb
```

### Analyze Results

View experiment metrics and visualizations:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Project Structure

```text
multi-llm-collaborative-debate-system/
├── main.py                 # Entry point and experiment loop
├── requirements.txt        # Python dependencies
├── .env                    # API keys
├── src/
│   ├── agents.py          # GPT wrapper + persona definitions
│   ├── models.py          # Pydantic schemas for structured outputs
│   └── orchestrator.py    # Stage functions (solve, critique, refine, judge)
├── data/
│   ├── problems.json      # 25 curated reasoning problems
│   └── results_log.json   # Experiment results (generated)
├── notebooks/
    ├── analysis.ipynb     # Results visualization
    └── demo_playground.ipynb  # Interactive single-run testing
```

## Problem Dataset

The system is evaluated on 25 curated problems across multiple categories:

| Category | Count | Examples |
| --- | --- | --- |
| Logic | 7 | Three Gods Puzzle, Cheryl's Birthday, 100 Prisoners |
| Math | 7 | Power towers, Factorials, Binomial coefficients |
| Physics | 4 | Ladder friction, Escape velocity, Inclined planes |
| Probability | 4 | Monty Hall, Meeting problem, Dice sums |
| Game Theory | 3 | Vickrey auction, Prisoner's dilemma |

## Technical Details

- **Model**: `gpt-4o-mini` (cost-effective, suitable for structured outputs)
- **Structured Outputs**: OpenAI's `response_format` with Pydantic models
- **Retry Logic**: 3 attempts with exponential backoff (2-10 seconds)
- **Grading**: LLM-as-a-Judge comparing final answer vs ground truth

## Output Format

Results are saved to `data/results_log.json` with the following structure:

```json
{
  "problem_id": 1,
  "category": "Logic",
  "difficulty": "Hard",
  "judge_id": "C",
  "solver_ids": ["A", "B", "D"],
  "initial_solutions": { ... },
  "reviews": { ... },
  "refined_solutions": { ... },
  "verdict": {
    "best_solver_id": "A",
    "final_answer_to_user": "..."
  },
  "evaluation": {
    "is_correct": true,
    "reasoning": "..."
  }
}
```

## Metrics

The analysis notebook computes:

- **Overall Accuracy**: Percentage of correct final answers
- **Judge vs Majority Vote**: Compare judge selection to consensus
- **Collaboration Bonus**: How often refinement improved answers
- **Category Breakdown**: Accuracy by problem type (Math, Logic, Physics)
