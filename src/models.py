from typing import Literal

from pydantic import BaseModel, Field


class RolePreference(BaseModel):
    agent_id: str
    role_priority: Literal["Solver", "Judge"]
    confidence: float = Field(description="Between 0.0 and 1.0")
    reasoning: str


class Solution(BaseModel):
    solution_text: str = Field(description="Step-by-step reasoning")
    final_answer: str = Field(
        description="The concise final answer (e.g., '42', 'Option B')"
    )


class CritiqueError(BaseModel):
    location: str = Field(description="Where the error occurred (e.g., 'Step 3')")
    description: str
    severity: Literal["minor", "critical"]


class PeerReview(BaseModel):
    reviewer_id: str
    target_solver_id: str
    strengths: list[str]
    weaknesses: list[str]
    errors: list[CritiqueError]
    score: int = Field(description="Quality score out of 10")


class RefinedSolution(BaseModel):
    changes_made: str = Field(description="Summary of changes based on feedback")
    solution_text: str = Field(description="The improved step-by-step reasoning")
    final_answer: str


class JudgeVerdict(BaseModel):
    best_solver_id: str
    rationale: str
    final_answer_to_user: str
