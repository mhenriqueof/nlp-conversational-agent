"""
Responsible for extracting user information from conversations and updating the user profile
automatically using the LLM.
"""

import json
from agent.llm.client import LLMClient
from agent.profile.schema import UserProfile


class ProfileUpdater:
    """
    Uses the LLM to extract structured user information from conversations and merges it into
    the existing user profile.

    Attributes:
        _llm: Private LLMClient instance for extraction requests.
    """

    def __init__(self) -> None:
        """
        Initializes the ProfileUpdater with an LLMClient instance.
        """
        self._llm = LLMClient()

    def extract_and_update(
        self, user_message: str, profile: UserProfile
    ) -> UserProfile:
        """
        Extracts from information from a message and updates the profile.

        Args:
            user_message: The current message from the user.
            profile: The current UserProfile instance to update.

        Returns:
            An updated UserProfile instance.
        """
        extraction_prompt = f"""
You are an information extraction assistant.
Analyze the user message below and extract any personal information mentioned. Return ONLY
a valid JSON object with these exact keys:
- "name": string or null
- "interests": list of strings or null
- "goals": list of strings or null
- "values": list of strings or null

Rules:
- Only extract information explicitly mentioned in the message.
- If nothing relevant is found for a field, use null.
- Never invent or assume information.
- Return ONLY the JSON object, no explanation, no markdown.

User message: "{user_message}"
"""

        raw = self._llm.generate_response(extraction_prompt)

        try:
            # Clean possible markdown fences
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("````")
                .strip()
            )
            extracted = json.loads(cleaned)
        except json.JSONDecodeError:
            return profile

        # Merge extracted info into existing profile
        if extracted.get("name"):
            profile.name = extracted["name"]
        if extracted.get("interests"):
            profile.interests = list(set(profile.interests + extracted["interests"]))
        if extracted.get("goals"):
            profile.goals = list(set(profile.goals + extracted["goals"]))
        if extracted.get("values"):
            profile.values = list(set(profile.values + extracted["values"]))

        profile.save()
        return profile
