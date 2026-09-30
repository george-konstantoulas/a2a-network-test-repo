from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):

    model_config = SettingsConfigDict(
        env_file =".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
    bucket_name : str
    fine_tune_dataset_path: str
    output_path: str
    base_llm: str
    fine_tuned_endpoint : str
    project_id : str
    location : str

config = Config()