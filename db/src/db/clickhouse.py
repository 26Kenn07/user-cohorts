import logging
from clickhouse_connect import get_async_client

from utils.config import settings

logger = logging.getLogger(__name__)


async def get_user_data_by_brand_id(
    brand_ids: str,
    start_date: str,
    end_date: str,
    limit: int = 50000,
    offset: int = 0,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    client = await get_async_client(
        host=settings.clickhouse.host,
        port=settings.clickhouse.port,
        username=settings.clickhouse.user_name,
        password=settings.clickhouse.password,
        database=settings.clickhouse.database
    )

    query = f"""
    select
        brand_id,
        multiIf(
            context_device_advertising_id <> '', context_device_advertising_id,
            identity_id
        ) as user_id,
        multiIf(
            context_device_advertising_id <> '', 'ad_id',
            'id_id'
        ) as identity_type,
        multiIf(
            video_id <> '', replaceAll(video_id, '"', ''),
            replaceAll(content_id, '"', '')
        ) as video_id,
        report_date,
        countIf(event = 'video_watched') as views,
        multiIf(countIf(event = 'video_sparked') > 0, 1, 0) as likes,
        countIf(event = 'video_shared') as shares,
        countIf(event = 'commented_on_video') as comments,
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
    from genuin_events_logs_001
    where brand_id in {brand_ids}
        and event IN ('video_impression', 'video_watched', 'video_sparked', 'video_shared', 'commented_on_video')
        and report_date BETWEEN '{start_date}' AND '{end_date}'
    group by brand_id, user_id, identity_type, video_id, report_date
    HAVING views >= 1
    order by user_id
    limit {limit}
    offset {offset}
    """

    try:
        result = await client.query_df(query)
        logger.info(f"Fetched {len(result)} rows from {start_date} to {end_date}")
        return result

    except Exception as e:
        logger.error(f"Failed to get data from clickhouse: {e}")
        raise e

    finally:
        await client.close()
