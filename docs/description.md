# Multi-LLM Collaborative Debate System

## Assignment Overview

Build a debate system where **three LLMs solve problems independently**, then cross-evaluate each other's solutions through structured critique. After mutual refinement based on peer feedback, a **fourth LLM judges** all final solutions and returns the best answer. This system combats hallucination through diverse perspectives and adversarial review.

---

## Phase 1: Problem Dataset Construction

**Goal:** Construct a dataset of **25 challenging problems**.

### Suggested Problem Categories

1. **Mathematical/Logical Reasoning**
    - Complex combinatorics, probability puzzles, number theory proofs.
    - Problems where LLMs commonly make calculation errors or logical leaps.
    - _Example:_ "In how many ways can you tile a 3×8 rectangle with 2×1 dominoes?"

2. **Physics & Scientific Reasoning**
    - Multi-step physics problems requiring formula application and unit analysis.
    - Counterintuitive scenarios (Monty Hall-style physics problems).
    - _Example:_ "A ladder leans against a frictionless wall. Derive the minimum coefficient of friction needed with the ground to prevent slipping."

3. **Logic Puzzles & Constraint Satisfaction**
    - Multi-agent reasoning problems (knights and knaves, truth-tellers).
    - Constraint satisfaction with interdependent rules.
    - _Example:_ "Five people of different nationalities live in five colored houses. Given 15 clues about their pets, drinks, and cigarette brands, who owns the fish?"

4. **Strategic Game Theory**
    - Optimal strategy derivation in games with incomplete information.
    - Backward induction problems, Nash equilibria calculations.
    - _Example:_ "In a two-player auction where bids are sealed and highest bidder pays the second-highest bid, what's the optimal bidding strategy?"

### Requirements

- Each problem must have a **verifiable correct answer**.
- Problems should be challenging enough that single LLM attempts often fail.

---

## Phase 2: System Implementation

### The Four Roles

**Three Equal Solvers and 1 Final Judge**

1. **Solver 1:** Independent solution with reasoning.
2. **Solver 2:** Independent solution with reasoning.
3. **Solver 3:** Independent solution with reasoning.
4. **Final Judge:** Evaluates all refined solutions and picks the best.

### Workflow

#### Stage 0: Role Assignment

Give each LLM the original question and ask them which role they think will be best for them for this question. Each of the 4 LLMs (GPT-4, Claude, Gemini, Grok) self-assesses:

```json
{
  "role_preferences": ["Solver", "Judge"],
  "confidence_by_role": {
    "Solver_1": 0.85,
    "Judge": 0.75
  },
  "reasoning": "I should be Solver because I'm strong at mathematical reasoning..."
}
```

_(Or use any other format that helps better assign roles)._

#### Stage 0.5: Algorithmic Role Assignment

After receiving results, use a deterministic algorithm to choose the final role distribution.

#### Stage 1: Independent Solution Generation

All three Solvers independently generate complete solutions with step-by-step reasoning.

- **Note:** No communication between Solvers at this stage.

#### Stage 2: Peer Review Round

Each Solver evaluates the other two solutions using structured feedback. Each Solver produces 2 reviews (one for each peer).

**Example Output Structure:**

```json
{
  "solution_id": "solver_2",
  "evaluation": {
    "strengths": ["Clear step 1-3", "Correct formula application"],
    "weaknesses": [
      "Step 5 makes unjustified leap",
      "Didn't consider edge case X"
    ],
    "errors": [
      {
        "location": "Step 5",
        "error_type": "logical_error",
        "description": "Claims X implies Y but this is false when Z...",
        "severity": "critical"
      }
    ],
    "suggested_changes": [
      "Reconsider step 5 with counterexample...",
      "Add verification for case when n=0"
    ]
  },
  "overall_assessment": "promising_but_flawed"
}
```

#### Stage 3: Refinement Based on Feedback

Each Solver receives 2 reviews from their peers and must:

1. Address each critique explicitly.
2. Defend their reasoning if critiques are wrong.
3. Revise their solution incorporating valid feedback.
4. Produce a refined final solution.

**Example Output Structure:**

```json
{
  "changes_made": [
    {
      "critique": "Step 5 was wrong",
      "response": "Fixed by...",
      "accepted": true
    },
    {
      "critique": "Missing edge case",
      "response": "This case doesn't apply because...",
      "accepted": false
    }
  ],
  "refined_solution": "...",
  "refined_answer": "...",
  "confidence": 0.9
}
```

#### Stage 4: Final Judgment

The Judge receives:

- All three original solutions.
- All peer reviews.
- All three refined solutions.

The Judge must produce:

```json
{
  "winner": "solver_1",
  "confidence": 0.85,
  "reasoning": "Solver 1's solution is strongest because..."
}
```

Depending on the winner, copy the answer and return it to the user.

---

## Phase 3: Evaluation and Analysis

### Quantitative Metrics

**System-Level Performance**

- **Overall Accuracy:** % of problems solved correctly by final answer.
- **Improvement Rate:** % of problems where refinement improved initial answers.
- **Consensus Rate:** % of problems where all 3 Solvers reached the same answer.
- **Judge Accuracy:** When Solvers disagree, does the Judge pick the correct one?

**Comparison to Baseline**

- **Single-LLM Baseline:** Accuracy of "just ask GPT-4/Claude/Gemini/Grok once".
- **Simple Voting Baseline:** 3 independent solutions, pick majority answer.
- **Your System:** Full debate with refinement.

---

## Deliverables

- A project that is as close as possible to a production-ready system.
- Code decomposition and structure are up to you.
- **Generated plots of evaluation results are A MUST!**

## Submission Format

A GitHub repository link. The repository must contain:

1. Notebooks.
2. Model artifacts.
3. A README.
4. Instructions on how to run the code.

> **⚠️ IMPORTANT INFORMATION:**
> Use a **free model** for all four roles (making 4 different API calls to the same free model with different parameters/system prompts).
