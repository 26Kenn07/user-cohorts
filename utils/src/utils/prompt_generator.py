import logging
import os
from dotenv import load_dotenv
from typing import Any

import anthropic
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)

_anthropic_client: anthropic.Anthropic | None = None
_gemini_client: genai.Client | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


def generate_cohort_label(
    cohort_id: int,
    video_cohort_map: dict[str, int],
    videos: list[dict[str, Any]],
    n_samples: int = 15,
) -> tuple[str, str]:
    """
    Uses Claude to produce a short label and description for a cohort
    based on actual video content (ai description → description → transcript),
    not keywords.
    Returns (label, description).
    """
    cohort_video_ids = {vid for vid, cid in video_cohort_map.items() if cid == cohort_id}
    cohort_videos = [v for v in videos if v["video_id"] in cohort_video_ids]

    snippets: list[str] = []
    for v in cohort_videos[:n_samples]:
        text = (
            v.get("video_gen_description") or
            v.get("description_text") or
            v.get("transcript") or ""
        ).strip()[:300]
        if text:
            snippets.append(text)

    if not snippets:
        return (f"Cohort {cohort_id}", "")

    content = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
    prompt = (
        "You are categorizing content clusters for a short video platform.\n\n"
        f"Here are {len(snippets)} video descriptions from one content cluster:\n\n"
        f"{content}\n\n"
        "Provide:\n"
        "1. A short label (2-4 words) capturing the theme\n"
        "2. A one-sentence description (max 12 words)\n\n"
        "Respond in exactly this format:\n"
        "Label: <label>\n"
        "Description: <description>"
    )
 
    response = _get_gemini_client().models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    label, description = f"Cohort {cohort_id}", ""
    for line in response.text.strip().splitlines():
        if line.startswith("Label:"):
            label = line.removeprefix("Label:").strip()
        elif line.startswith("Description:"):
            description = line.removeprefix("Description:").strip()

    logger.info(f"Cohort {cohort_id} → '{label}': {description}")
    return label, description


def generate_all_labels(
    cohort_profiles: list[dict[str, Any]],
    video_cohort_map: dict[str, int],
    videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enriches each cohort profile with a Claude-generated label and description."""
    results = []
    for profile in cohort_profiles:
        label, description = generate_cohort_label(
            cohort_id=profile["cohort_id"],
            video_cohort_map=video_cohort_map,
            videos=videos,
        )
        results.append({**profile, "label": label, "description": description})
    return results


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
    if not top_keywords:
        logger.warning(f"Cohort {cohort_id} has no keywords — skipping prompt generation")
        return []

    client = _get_anthropic_client()

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
