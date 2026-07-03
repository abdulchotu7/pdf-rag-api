from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLM(ABC):
    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        ...

    @abstractmethod
    async def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        ...
        yield
