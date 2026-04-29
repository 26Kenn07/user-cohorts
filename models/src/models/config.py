from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=True
)

class ClickHouseConfig(BaseSettings):
    model_config = MODEL_CONFIG
    
    host: str = Field(..., alias="CLICKHOUSE_HOST")
    port: str = Field(..., alias="CLICKHOUSE_PORT")
    user_name: str = Field(..., alias="CLICKHOUSE_USER")
    password: str = Field(..., alias="CLICKHOUSE_PASSWORD")
    database: str = Field(..., alias="CLICKHOUSE_DATABASE")
    
class OpenSearchConfig(BaseSettings):
    model_config = MODEL_CONFIG
    
    url: str = Field(default="http://localhost:9200", alias="OS_URL")
    user_name: str = Field(default="", alias="OS_USER_NAME")
    password: str = Field(default="", alias="OS_PASS")
    os_verify: bool = Field(default=False, alias="OS_VERIFY")
    
    
    
class AppConfig(BaseSettings):
    model_config = MODEL_CONFIG
    
    env: str = Field(..., alias="ENV")
    log_level: str = Field(..., alias="LOG_LEVEL")
    
    
__all__ = [
    "ClickHouseConfig",
    "AppConfig"
]