"""
Agent module for the Multi-LLM Collaborative Debate System.

Contains persona definitions and the GPT API wrapper with retry logic.
"""

import logging
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "gpt-4o-mini"

# Agent Personas - distinct system prompts to prevent mode collapse
PERSONAS: dict[str, str] = {
    "A": (
        "You are a rigid logician and scientist. You prioritize formal proofs, "
        "step-by-step derivation, and edge cases. You are skeptical of intuition "
        "and require verification for every claim."
    ),
    "B": (
        "You are a lateral thinker. You look for alternative interpretations, "
        "trick wording, or creative solutions that standard logic might miss. "
        "You value outside-the-box reasoning."
    ),
    "C": (
        "You are a pragmatic engineer. You focus on probability, heuristics, "
        "and real-world constraints. You check if the answer 'makes sense' "
        "intuitively before diving into math."
    ),
    "D": (
        "You are a balanced synthesizer. You strive for clarity and consensus. "
        "You try to bridge the gap between abstract logic and practical application."
    ),
}

T = TypeVar("T", bound=BaseModel)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(
        f"API call failed, retrying (attempt {retry_state.attempt_number})..."
    ),
)
def call_gpt(
    client: OpenAI,
    agent_id: str,
    user_prompt: str,
    response_model: type[T],
    system_prompt_override: str | None = None,
) -> T:
    """
    Call GPT-4o-mini with structured output parsing.

    Args:
        client: OpenAI client instance.
        agent_id: Agent identifier (A, B, C, or D) to select persona.
        user_prompt: The user message/prompt to send.
        response_model: Pydantic model class for structured output parsing.
        system_prompt_override: Optional override for the system prompt.
            If provided, replaces the persona-based system prompt.

    Returns:
        Parsed response as an instance of the response_model.

    Raises:
        ValueError: If agent_id is not found in PERSONAS.
        Exception: If API call fails after all retry attempts.
    """
    if agent_id not in PERSONAS and system_prompt_override is None:
        raise ValueError(
            f"Unknown agent_id: {agent_id}. Must be one of {list(PERSONAS.keys())}"
        )

    system_prompt = (
        system_prompt_override if system_prompt_override else PERSONAS[agent_id]
    )

    logger.debug(f"Calling GPT for agent {agent_id} with model {MODEL_NAME}")

    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse response from GPT")

    return parsed
