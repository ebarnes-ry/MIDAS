import ssl

import httpx
import openai
import truststore
from marker.services.openai import OpenAIService


class TruststoreOpenAIService(OpenAIService):
    """Marker OpenAI service variant that uses the operating system trust store."""

    def get_client(self) -> openai.OpenAI:
        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        http_client = httpx.Client(verify=ssl_context, timeout=self.timeout)
        return openai.OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            http_client=http_client,
        )
