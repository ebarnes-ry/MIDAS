from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import BaseModel, ConfigDict

from src.models.providers.base import ChatRequest
from src.models.providers.openai_sdk import OpenAIProvider


class StrictTestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


def test_openai_schema_response_format_uses_strict_true():
    provider = OpenAIProvider(api_key="test-key")
    provider.client = Mock()
    provider.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"answer":"x = 1"}'),
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=None,
    )

    response = provider.chat(
        ChatRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Solve x + 1 = 2"}],
            params={},
            schema=StrictTestSchema,
        )
    )

    call_kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["strict"] is True
    assert (
        call_kwargs["response_format"]["json_schema"]["schema"]
        == StrictTestSchema.model_json_schema()
    )
    assert response.parsed == StrictTestSchema(answer="x = 1")
