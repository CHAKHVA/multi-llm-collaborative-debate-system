"""
Orchestrator module for the Multi-LLM Collaborative Debate System.

Contains the stage functions that drive the debate workflow.
"""

import logging

from openai import OpenAI

from src.agents import call_gpt
from src.models import (
    EvaluationResult,
    JudgeVerdict,
    PeerReview,
    RefinedSolution,
    RolePreference,
    Solution,
)

logger = logging.getLogger(__name__)


def get_role_preference(client: OpenAI, agent_id: str, question: str) -> RolePreference:
    """
    Stage 0: Get an agent's role preference (Solver vs Judge).

    Args:
        client: OpenAI client instance.
        agent_id: Agent identifier (A, B, C, or D).
        question: The problem question to analyze.

    Returns:
        RolePreference indicating the agent's preferred role and confidence.
    """
    prompt = f"""Analyze the following problem and decide whether you would be better suited as a Solver or a Judge.

**Problem:**
{question}

As a **Solver**, you will:
- Generate an independent solution to the problem
- Receive critiques from other solvers
- Refine your solution based on feedback

As a **Judge**, you will:
- Observe all solutions and critiques
- Evaluate the quality of reasoning
- Select the best final answer

Consider your strengths and the nature of this problem. Provide your role preference with a confidence score (0.0 to 1.0) and reasoning.

Your agent_id is: {agent_id}"""

    return call_gpt(client, agent_id, prompt, RolePreference)


def generate_solution(client: OpenAI, agent_id: str, question: str) -> Solution:
    """
    Stage 1: Generate an independent solution to the problem.

    Args:
        client: OpenAI client instance.
        agent_id: Agent identifier (A, B, C, or D).
        question: The problem question to solve.

    Returns:
        Solution containing step-by-step reasoning and final answer.
    """
    prompt = f"""Solve the following problem step by step.

**Problem:**
{question}

Provide your complete reasoning process and a clear final answer. Be thorough in your analysis."""

    return call_gpt(client, agent_id, prompt, Solution)


def generate_critique(
    client: OpenAI,
    reviewer_id: str,
    target_solver_id: str,
    question: str,
    solution: Solution,
) -> PeerReview:
    """
    Stage 2: Generate a peer review critique of another solver's solution.

    Args:
        client: OpenAI client instance.
        reviewer_id: Agent ID of the reviewer.
        target_solver_id: Agent ID of the solver being reviewed.
        question: The original problem question.
        solution: The solution to critique.

    Returns:
        PeerReview containing strengths, weaknesses, errors, and score.
    """
    prompt = f"""Review the following solution to the given problem. Provide a thorough critique.

**Problem:**
{question}

**Solution by Solver {target_solver_id}:**
{solution.solution_text}

**Their Final Answer:** {solution.final_answer}

Analyze this solution carefully:
1. Identify strengths in the reasoning
2. Identify weaknesses or gaps
3. Point out any specific errors (note the location, e.g., "Step 3")
4. Assign a quality score out of 10

Your reviewer_id is: {reviewer_id}
The target_solver_id is: {target_solver_id}"""

    return call_gpt(client, reviewer_id, prompt, PeerReview)


def refine_solution(
    client: OpenAI,
    agent_id: str,
    question: str,
    original_solution: Solution,
    reviews: list[PeerReview],
) -> RefinedSolution:
    """
    Stage 3: Refine solution based on peer review feedback.

    Args:
        client: OpenAI client instance.
        agent_id: Agent identifier (A, B, C, or D).
        question: The original problem question.
        original_solution: The solver's initial solution.
        reviews: List of peer reviews received.

    Returns:
        RefinedSolution with improvements based on feedback.
    """
    reviews_text = ""
    for i, review in enumerate(reviews, 1):
        reviews_text += f"""
**Review {i} (from Reviewer {review.reviewer_id}):**
- Strengths: {", ".join(review.strengths) if review.strengths else "None listed"}
- Weaknesses: {", ".join(review.weaknesses) if review.weaknesses else "None listed"}
- Errors: {"; ".join(f"{e.location}: {e.description} ({e.severity})" for e in review.errors) if review.errors else "None found"}
- Score: {review.score}/10
"""

    prompt = f"""Refine your solution based on the peer feedback you received.

**Original Problem:**
{question}

**Your Original Solution:**
{original_solution.solution_text}

**Your Original Answer:** {original_solution.final_answer}

**Peer Reviews Received:**
{reviews_text}

Address the critiques, fix any errors identified, and improve your solution. Clearly state what changes you made."""

    return call_gpt(client, agent_id, prompt, RefinedSolution)


def judge_verdict(
    client: OpenAI,
    judge_id: str,
    question: str,
    solver_ids: list[str],
    initial_solutions: dict[str, Solution],
    reviews: dict[str, list[PeerReview]],
    refined_solutions: dict[str, RefinedSolution],
) -> JudgeVerdict:
    """
    Stage 4: Judge evaluates all solutions and selects the winner.

    Args:
        client: OpenAI client instance.
        judge_id: Agent ID of the judge.
        question: The original problem question.
        solver_ids: List of solver agent IDs.
        initial_solutions: Dict mapping solver_id to their initial Solution.
        reviews: Dict mapping solver_id to list of PeerReviews they received.
        refined_solutions: Dict mapping solver_id to their RefinedSolution.

    Returns:
        JudgeVerdict with the winning solver and final answer.
    """
    debate_history = ""
    for solver_id in solver_ids:
        initial = initial_solutions[solver_id]
        refined = refined_solutions[solver_id]
        solver_reviews = reviews[solver_id]

        debate_history += f"""
=== SOLVER {solver_id} ===

**Initial Solution:**
{initial.solution_text}
**Initial Answer:** {initial.final_answer}

**Reviews Received:**
"""
        for review in solver_reviews:
            debate_history += f"""
- From Reviewer {review.reviewer_id}: Score {review.score}/10
  Strengths: {", ".join(review.strengths) if review.strengths else "None"}
  Weaknesses: {", ".join(review.weaknesses) if review.weaknesses else "None"}
  Errors: {len(review.errors)} identified
"""

        debate_history += f"""
**Refined Solution:**
{refined.solution_text}
**Refined Answer:** {refined.final_answer}
**Changes Made:** {refined.changes_made}

"""

    judge_system_prompt = (
        "You are an impartial judge evaluating a multi-agent debate. "
        "Your role is to carefully analyze all solutions, consider the critiques made, "
        "and select the solver with the best final answer. Focus on correctness, "
        "reasoning quality, and how well each solver addressed feedback."
    )

    prompt = f"""Evaluate the following debate and select the best solver.

**Problem:**
{question}

**Debate History:**
{debate_history}

Select the solver with the best final answer. Provide your rationale and state the final answer to present to the user."""

    return call_gpt(
        client,
        judge_id,
        prompt,
        JudgeVerdict,
        system_prompt_override=judge_system_prompt,
    )


def grade_answer(
    client: OpenAI,
    question: str,
    ground_truth: str,
    final_answer: str,
) -> EvaluationResult:
    """
    Grade the final answer against the ground truth using LLM-as-a-Judge.

    Args:
        client: OpenAI client instance.
        question: The original problem question.
        ground_truth: The correct answer.
        final_answer: The system's final answer to evaluate.

    Returns:
        EvaluationResult indicating correctness and reasoning.
    """
    grader_system_prompt = (
        "You are an expert grader evaluating answers to complex problems. "
        "Your job is to determine if the given answer is correct by comparing it "
        "to the ground truth. Be fair but rigorous. Consider semantic equivalence - "
        "answers may be phrased differently but still be correct."
    )

    prompt = f"""Evaluate whether the given answer is correct.

**Problem:**
{question}

**Ground Truth (Correct Answer):**
{ground_truth}

**Answer to Evaluate:**
{final_answer}

Determine if the answer is correct. Consider:
- Semantic equivalence (different phrasing, same meaning)
- Mathematical equivalence (e.g., "1/2" vs "0.5")
- Partial credit is NOT allowed - the answer is either correct or incorrect

Provide your reasoning and final verdict."""

    return call_gpt(
        client,
        "D",  # Use agent D (balanced synthesizer) for grading
        prompt,
        EvaluationResult,
        system_prompt_override=grader_system_prompt,
    )
