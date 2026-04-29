import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def generate_prompts_for_cohort(
    cohort_id: int,
    top_keywords: list[str],
    user_count: int,
    n_prompts: int = 5,
) -> list[str]:
    """
    Uses Claude to generate prompt suggestions for a cohort
    based on their top engagement keywords.
    """
    return 
    if not top_keywords:
        logger.warning(f"Cohort {cohort_id} has no keywords — skipping prompt generation")
        return []

    client = _get_client()

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"""You are helping generate video search prompts for a group of {user_count} users on a short video platform.

These users have shown strong engagement with videos related to these topics:
{", ".join(top_keywords)}

Generate exactly {n_prompts} short, natural search prompts that these users would likely type to find videos they enjoy.

Rules:
- Each prompt should be 3-8 words
- Make them specific to the topics above
- Write them as a user would naturally search
- Return only the prompts, one per line, no numbering or extra text""",
            }
        ],
    )

    raw = message.content[0].text.strip()
    prompts = [line.strip() for line in raw.splitlines() if line.strip()][:n_prompts]
    logger.info(f"Cohort {cohort_id} prompts: {prompts}")
    return prompts


def generate_all_prompts(
    cohort_profiles: list[dict[str, Any]],
    n_prompts: int = 5,
) -> list[dict[str, Any]]:
    """
    Generates prompts for all cohorts and returns enriched profiles.
    """
    results = []
    for profile in cohort_profiles:
        prompts = generate_prompts_for_cohort(
            cohort_id=profile["cohort_id"],
            top_keywords=profile["top_keywords"],
            user_count=profile["user_count"],
            n_prompts=n_prompts,
        )
        results.append({**profile, "prompts": prompts})

    return results
