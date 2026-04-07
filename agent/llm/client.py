"""
Responsible for communicating with the Hugging Face Inference API.
"""

import os

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from agent.llm.prompt import build_prompt
from agent.profile.schema import UserProfile

load_dotenv()


class LLMClient:
    """
    Client for communicating with the Hugging Face Inference API.
    Handles authentication and response generation using chat completion models.

    Attributes:
        _client: Private instance of InferenceClient authenticated with HF token.
    """

    def __init__(self):
        """
        Initializes the LLMClient by loading the HF token from evironment variables and creating
        an authenticated InferenceClient instance.

        Raises:
            ValueError: If HF token is not found in environment variables.
        """
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("'HF_TOKEN' not found. Check .env file.")
        self._client = InferenceClient(token=token)

    def generate_response(
        self,
        user_message: str,
        memory_context: list[str] = [],
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        max_new_tokens: int = 512,
        profile: UserProfile | None = None,
    ) -> str:
        """
        Generates a response from the LLM given a user message, optional memory context
        and optional user profile.

        Args:
            user_message: The current message from the user.
            memory_context: List of relevant past interactions retrieved from memory.
            model: HF model ID to use for inference.
            max_new_tokens: Maximum number of tokens to generate.
            profile: The current UserProfile instance for personalized responses.

        Returns:
            The generated response as a string.
        """
        prompt = build_prompt(user_message, memory_context, profile)

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()  # type: ignore
