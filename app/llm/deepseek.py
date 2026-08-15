from typing import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import LLMUnavailableError
from app.core.logging import get_logger
from app.llm.base import BaseLLM

logger = get_logger("deepseek")

_DISABLE_THINKING = {"thinking": {"type": "disabled"}}


class DeepSeekProvider(BaseLLM):
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    async def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.app_timeout,
            )
        return self._client

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=settings.deepseek_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=_DISABLE_THINKING,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("DeepSeek chat failed", exc_info=exc)
            raise LLMUnavailableError(
                f"DeepSeek LLM request failed: {exc}"
            ) from exc

    async def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        try:
            client = await self._get_client()
            stream = await client.chat.completions.create(
                model=settings.deepseek_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body=_DISABLE_THINKING,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as exc:
            logger.error("DeepSeek streaming chat failed", exc_info=exc)
            raise LLMUnavailableError(
                f"DeepSeek streaming request failed: {exc}"
            ) from exc
