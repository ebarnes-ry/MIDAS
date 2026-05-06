import os
from .openai_sdk import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """
    Thin wrapper around OpenAIProvider pointed at Groq's inference API.
    Groq is OpenAI-compatible, so we inherit all retry logic, error handling,
    structured output parsing, and image support from the existing provider.
    """

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            **kwargs,
        )
