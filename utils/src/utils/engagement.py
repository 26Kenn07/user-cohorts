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
        + df["views"].clip(upper=3) * 0.5          # cap at 3 to reduce noise from loops
        + df["likes"]       * 3.0
        + df["shares"]      * 5.0
        + df["comments"]    * 3.0
        + (df["link_clicks"] if "link_clicks" in df.columns else 0) * 4.0
    )

    # Normalize per brand so a single super-engaged user on one brand
    # doesn't collapse scores for all other brands to near zero
    df["score"] = df.groupby("brand_id")["score"].transform(
        lambda g: g / g.max() if g.max() > 0 else g
    )

    cols = ["user_id", "identity_type", "video_id", "brand_id",
            "watch_percentage", "views", "likes", "shares", "comments", "score"]
    if "link_clicks" in df.columns:
        cols.insert(-1, "link_clicks")
    return df[cols]
