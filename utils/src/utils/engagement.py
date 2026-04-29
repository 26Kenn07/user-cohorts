import pandas as pd


def compute_engagement_scores(user_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a single engagement score per (user_id, video_id) pair.

    Repeated views are counted but diminish in value — watching twice
    is better than once but not twice as good.
    """
    df = user_df.copy()

    df["score"] = (
        (df["watch_percentage"] / 100) * 1.0
        + df["views"].clip(upper=3) * 0.5   # cap at 3 to reduce noise from loops
        + df["likes"]    * 3.0
        + df["shares"]   * 5.0
        + df["comments"] * 3.0
    )

    # Normalize score to [0, 1] per brand so scores are comparable across brands
    max_score = df["score"].max()
    if max_score > 0:
        df["score"] = df["score"] / max_score

    return df[["user_id", "identity_type", "video_id", "brand_id",
               "watch_percentage", "views", "likes", "shares", "comments", "score"]]
