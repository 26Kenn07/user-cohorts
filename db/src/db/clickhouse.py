import logging
from clickhouse_connect import get_async_client

from utils.config import settings

logger = logging.getLogger(__name__)

async def get_urls(limit: int, offset: int):
    client = await get_async_client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        username=settings.clickhouse.user_name,
        password=settings.clickhouse.password,
        database=settings.clickhouse.database,
        send_receive_timeout=18000,
    )

    query = f"""
        WITH agg AS (
        SELECT
            brand_id,
            multiIf(
                context_device_advertising_id <> '', context_device_advertising_id,
                identity_id
            ) AS user_id,
            url,
            multiIf(
                context_device_advertising_id <> '', 'ad_id',
                'id_id'
            ) AS identity_type,
            multiIf(
                video_id <> '', replaceAll(video_id, '"', ''),
                replaceAll(content_id, '"', '')
            ) AS video_id,
            report_date,
            countIf(event = 'video_watched')                                                   AS views,
            multiIf(countIf(event = 'video_sparked') > 0, 1, 0)                               AS likes,
            countIf(event = 'video_shared')                                                    AS shares,
            countIf(event = 'commented_on_video')                                              AS comments,
            countIf(event IN (
                'link_clicked', 'link_button_clicked', 'links_card_clicked',
                'link_cta_clicked', 'link_cta_button_clicked'
            ))                                                                                 AS link_clicks,
            ROUND(
                (
                    if(
                        sumIf(video_view_length, event = 'video_impression') > 0,
                        sumIf(video_view_length, event = 'video_impression'),
                        sumIf(video_view_length, event = 'video_watched')
                    ) * 100
                )
                /
                if(
                    sumIf(video_length, event = 'video_impression') > 0,
                    sumIf(video_length, event = 'video_impression'),
                    sumIf(video_length, event = 'video_watched')
                ),
                2
            ) AS watch_percentage
        FROM genuin_events_logs_001
        WHERE brand_id IN (1729, 2023, 2075, 2556, 2558, 2314, 2357, 2023, 2476, 2557, 2701, 2764, 2790, 2793, 2801, 2808, 3099)
            AND event IN (
                'video_impression', 'video_watched', 'video_sparked',
                'video_shared', 'commented_on_video',
                'link_clicked', 'link_button_clicked', 'links_card_clicked',
                'link_cta_clicked', 'link_cta_button_clicked'
            )
            AND report_date BETWEEN '2025-05-01' AND '2026-04-29'
            AND event_record_screen NOT IN ('view_embed', 'view_placement', '', 'feed', 'carousel')
        GROUP BY brand_id, user_id, url, identity_type, video_id, report_date
        HAVING views >= 1
        ),
        qualified_users AS (
        SELECT brand_id, user_id
        FROM agg
        GROUP BY brand_id, user_id
        HAVING count(*) >= 10
        )
        SELECT 
        DISTINCT(brand_id, url)
        FROM agg
        INNER JOIN qualified_users q
        ON agg.brand_id = q.brand_id AND agg.user_id = q.user_id
        order by brand_id, url
        limit {limit}
        offset {offset}
    """

    try:
        result = await client.query_df(query)
        logger.info(f"Fetched {len(result)} rows (offset={offset}) from {start_date} to {end_date}")
        return result

    except Exception as e:
        logger.error(f"Failed to get data from clickhouse: {e}")
        raise e

    finally:
        await client.close()


async def get_user_data_by_brand_id(
    brand_ids: str,
    start_date: str,
    end_date: str,
    min_events: int = 10,
    limit: int = 50000,
    offset: int = 0,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    client = await get_async_client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        username=settings.clickhouse.user_name,
        password=settings.clickhouse.password,
        database=settings.clickhouse.database,
        send_receive_timeout=18000,
    )

    query = f"""
    WITH agg AS (
        SELECT
            brand_id,
            multiIf(
                context_device_advertising_id <> '', context_device_advertising_id,
                identity_id
            ) AS user_id,
            multiIf(
                context_device_advertising_id <> '', 'ad_id',
                'id_id'
            ) AS identity_type,
            multiIf(
                video_id <> '', replaceAll(video_id, '"', ''),
                replaceAll(content_id, '"', '')
            ) AS video_id,
            report_date,
            countIf(event = 'video_watched')                                                   AS views,
            multiIf(countIf(event = 'video_sparked') > 0, 1, 0)                               AS likes,
            countIf(event = 'video_shared')                                                    AS shares,
            countIf(event = 'commented_on_video')                                              AS comments,
            countIf(event IN (
                'link_clicked', 'link_button_clicked', 'links_card_clicked',
                'link_cta_clicked', 'link_cta_button_clicked'
            ))                                                                                 AS link_clicks,
            ROUND(
                (
                    if(
                        sumIf(video_view_length, event = 'video_impression') > 0,
                        sumIf(video_view_length, event = 'video_impression'),
                        sumIf(video_view_length, event = 'video_watched')
                    ) * 100
                )
                /
                if(
                    sumIf(video_length, event = 'video_impression') > 0,
                    sumIf(video_length, event = 'video_impression'),
                    sumIf(video_length, event = 'video_watched')
                ),
                2
            ) AS watch_percentage
        FROM genuin_events_logs_001
        WHERE brand_id IN {brand_ids}
          AND event IN (
              'video_impression', 'video_watched', 'video_sparked',
              'video_shared', 'commented_on_video',
              'link_clicked', 'link_button_clicked', 'links_card_clicked',
              'link_cta_clicked', 'link_cta_button_clicked'
          )
          AND report_date BETWEEN '{start_date}' AND '{end_date}'
          AND event_record_screen NOT IN ('view_embed', 'view_placement', '', 'feed', 'carousel')
        GROUP BY brand_id, user_id, identity_type, video_id, report_date
        HAVING views >= 1
    ),
    qualified_users AS (
        SELECT brand_id, user_id
        FROM agg
        GROUP BY brand_id, user_id
        HAVING count(*) > {min_events}
    )
    SELECT agg.*
    FROM agg
    INNER JOIN qualified_users q
        ON agg.brand_id = q.brand_id AND agg.user_id = q.user_id
    ORDER BY agg.user_id
    LIMIT {limit} OFFSET {offset}
    """

    try:
        result = await client.query_df(query)
        logger.info(f"Fetched {len(result)} rows (offset={offset}) from {start_date} to {end_date}")
        return result

    except Exception as e:
        logger.error(f"Failed to get data from clickhouse: {e}")
        raise e

    finally:
        await client.close()

