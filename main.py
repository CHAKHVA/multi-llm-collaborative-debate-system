"""
Main entry point for the Multi-LLM Collaborative Debate System.

Orchestrates the full debate workflow: role assignment, solution generation,
peer review, refinement, judging, and grading.
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.agents import PERSONAS
from src.models import (
    EvaluationResult,
    JudgeVerdict,
    PeerReview,
    RefinedSolution,
    RolePreference,
    Solution,
)
from src.orchestrator import (
    generate_critique,
    generate_solution,
    get_role_preference,
    grade_answer,
    judge_verdict,
    refine_solution,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
PROBLEMS_PATH = DATA_DIR / "problems.json"
RESULTS_PATH = DATA_DIR / "results_log.json"


def load_problems(path: Path) -> list[dict]:
    """
    Load problems from JSON file.

    Args:
        path: Path to the problems JSON file.

    Returns:
        List of problem dictionaries.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_results(results: list[dict], path: Path) -> None:
    """
    Save results to JSON file.

    Args:
        results: List of result dictionaries.
        path: Path to save the results.
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


def assign_roles(
    preferences: list[RolePreference],
) -> tuple[str, list[str]]:
    """
    Assign roles based on agent preferences using weighted random selection.

    Args:
        preferences: List of RolePreference from all agents.

    Returns:
        Tuple of (judge_id, list of solver_ids).
    """
    # Find agents who prefer to be Judge
    judge_candidates = [p for p in preferences if p.role_priority == "Judge"]

    if judge_candidates:
        # Weighted random selection based on confidence
        weights = [p.confidence for p in judge_candidates]
        judge_pref = random.choices(judge_candidates, weights=weights, k=1)[0]
        judge_id = judge_pref.agent_id
    else:
        # If no one wants to be judge, pick randomly
        judge_pref = random.choice(preferences)
        judge_id = judge_pref.agent_id

    # Remaining agents become solvers
    solver_ids = [p.agent_id for p in preferences if p.agent_id != judge_id]

    logger.info(f"Role Assignment: Judge={judge_id}, Solvers={solver_ids}")
    return judge_id, solver_ids


def run_debate(
    client: OpenAI,
    problem: dict,
) -> dict:
    """
    Run the full debate workflow for a single problem.

    Args:
        client: OpenAI client instance.
        problem: Problem dictionary with 'question' and 'ground_truth'.

    Returns:
        Result dictionary with all debate data and evaluation.
    """
    question = problem["question"]
    ground_truth = problem["ground_truth"]
    problem_id = problem["id"]

    logger.info(f"=== Processing Problem {problem_id} ===")
    logger.info(f"Category: {problem.get('category', 'Unknown')}")

    # Stage 0: Role Assignment
    logger.info("Stage 0: Getting role preferences...")
    agent_ids = list(PERSONAS.keys())
    preferences: list[RolePreference] = []

    for agent_id in agent_ids:
        pref = get_role_preference(client, agent_id, question)
        preferences.append(pref)
        logger.info(
            f"  Agent {agent_id}: {pref.role_priority} (confidence: {pref.confidence:.2f})"
        )

    judge_id, solver_ids = assign_roles(preferences)

    # Stage 1: Independent Solutions
    logger.info("Stage 1: Generating independent solutions...")
    initial_solutions: dict[str, Solution] = {}

    for solver_id in solver_ids:
        solution = generate_solution(client, solver_id, question)
        initial_solutions[solver_id] = solution
        logger.info(f"  Solver {solver_id} answer: {solution.final_answer[:50]}...")

    # Stage 2: Peer Review (Round Robin)
    logger.info("Stage 2: Conducting peer reviews...")
    all_reviews: dict[str, list[PeerReview]] = {sid: [] for sid in solver_ids}

    for reviewer_id in solver_ids:
        for target_id in solver_ids:
            if reviewer_id != target_id:
                review = generate_critique(
                    client,
                    reviewer_id,
                    target_id,
                    question,
                    initial_solutions[target_id],
                )
                all_reviews[target_id].append(review)
                logger.info(
                    f"  {reviewer_id} reviewed {target_id}: Score {review.score}/10"
                )

    # Stage 3: Refinement
    logger.info("Stage 3: Refining solutions...")
    refined_solutions: dict[str, RefinedSolution] = {}

    for solver_id in solver_ids:
        refined = refine_solution(
            client,
            solver_id,
            question,
            initial_solutions[solver_id],
            all_reviews[solver_id],
        )
        refined_solutions[solver_id] = refined
        logger.info(
            f"  Solver {solver_id} refined answer: {refined.final_answer[:50]}..."
        )

    # Stage 4: Judge Verdict
    logger.info("Stage 4: Getting judge verdict...")
    verdict: JudgeVerdict = judge_verdict(
        client,
        judge_id,
        question,
        solver_ids,
        initial_solutions,
        all_reviews,
        refined_solutions,
    )
    logger.info(f"  Winner: Solver {verdict.best_solver_id}")
    logger.info(f"  Final Answer: {verdict.final_answer_to_user}")

    # Grading
    logger.info("Grading final answer...")
    evaluation: EvaluationResult = grade_answer(
        client, question, ground_truth, verdict.final_answer_to_user
    )
    logger.info(f"  Correct: {evaluation.is_correct}")

    # Compile result
    result = {
        "problem_id": problem_id,
        "category": problem.get("category"),
        "difficulty": problem.get("difficulty"),
        "question": question,
        "ground_truth": ground_truth,
        "role_preferences": [p.model_dump() for p in preferences],
        "judge_id": judge_id,
        "solver_ids": solver_ids,
        "initial_solutions": {k: v.model_dump() for k, v in initial_solutions.items()},
        "reviews": {k: [r.model_dump() for r in v] for k, v in all_reviews.items()},
        "refined_solutions": {k: v.model_dump() for k, v in refined_solutions.items()},
        "verdict": verdict.model_dump(),
        "evaluation": evaluation.model_dump(),
        "timestamp": datetime.now().isoformat(),
    }

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-LLM Collaborative Debate System"
    )
    parser.add_argument(
        "--test-id",
        type=int,
        help="Run only on a specific problem ID",
    )
    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI()

    # Load problems
    problems = load_problems(PROBLEMS_PATH)
    logger.info(f"Loaded {len(problems)} problems")

    # Filter to specific problem if requested
    if args.test_id is not None:
        problems = [p for p in problems if p["id"] == args.test_id]
        if not problems:
            logger.error(f"Problem with ID {args.test_id} not found")
            return
        logger.info(f"Running on problem ID {args.test_id} only")

    # Run debates
    results: list[dict] = []

    for problem in problems:
        try:
            result = run_debate(client, problem)
            results.append(result)

            # Save after each problem (incremental saving)
            save_results(results, RESULTS_PATH)

        except Exception as e:
            logger.error(f"Error processing problem {problem['id']}: {e}")
            results.append(
                {
                    "problem_id": problem["id"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    # Final summary
    correct_count = sum(
        1 for r in results if r.get("evaluation", {}).get("is_correct", False)
    )
    total_count = len([r for r in results if "evaluation" in r])

    logger.info("=== Final Summary ===")
    logger.info(f"Total problems processed: {len(results)}")
    logger.info(f"Correct answers: {correct_count}/{total_count}")
    if total_count > 0:
        logger.info(f"Accuracy: {correct_count / total_count * 100:.1f}%")


if __name__ == "__main__":
    main()
