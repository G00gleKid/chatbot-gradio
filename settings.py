from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    LLM_HTTP_PROXY: str

    GIGACHAT_SCOPE: str
    GIGACHAT_AUTH_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
