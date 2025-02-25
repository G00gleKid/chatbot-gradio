import httpx
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
from settings import settings


class LLMParamsDTO(BaseModel):
    timeout: int = 180
    temperature: float = 0.6
    max_tokens: int = 4096


class ProxyHttpxClient(httpx.Client):
    def __init__(self) -> None:
        super().__init__(proxy=settings.LLM_HTTP_PROXY)


class LangchainOpenAI(ChatOpenAI):
    def __init__(self, params: LLMParamsDTO | None = None) -> None:
        print(params)
        if not params:
            params = LLMParamsDTO()

        super().__init__(
            api_key=settings.OPENAI_API_KEY,  # type: ignore[arg-type]
            model="gpt-4o",
            http_client=ProxyHttpxClient(),
            **params.model_dump(),
        )


class LangchainGigaChat(GigaChat):
    def __init__(self, params: LLMParamsDTO | None = None) -> None:
        if not params:
            params = LLMParamsDTO()

        super().__init__(
            credentials=settings.GIGACHAT_AUTH_KEY,
            scope=settings.GIGACHAT_SCOPE,
            model="GigaChat",
            verify_ssl_certs=False,
            **params.model_dump(),
            streaming=True,
        )
