"""
Defines the user profile schema using Pydantic.
"""

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """
    Represents the agent's structured long-term model of the user.

    Attributes:
        name: The user's name.
        interests: List of topics, hobbies or activities the user enjoys.
        goals: List of objectives or aspirations the user has mentioned.
        values: List of principles or things the user cares about.
    """

    name: str = Field(default="", description="The user's name.")
    interests: list[str] = Field(
        default_factory=list, description="Topics, hobbies or activities the enjoys."
    )
    goals: list[str] = Field(
        default_factory=list,
        description="Objectives or aspirations the user has mentioned.",
    )
    values: list[str] = Field(
        default_factory=list, description="Principles or things the user cares about."
    )

    def to_prompt_string(self) -> str:
        """
        Converts the profile into a readable string for prompt injection.

        Returns:
            A formatted string summarizing the user profile.
        """
        lines = ["[User Profile]"]

        if self.name:
            lines.append(f"Name: {self.name}")
        if self.interests:
            lines.append(f"Interests: {', '.join(self.interests)}")
        if self.goals:
            lines.append(f"Goals: {', '.join(self.goals)}")
        if self.values:
            lines.append(f"Values: {', '.join(self.values)}")

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def save(self, path: str = "user_profile.json") -> None:
        """
        Saves the user profile to a JSON file.

        Args:
            path: Path to the JSON file. Defaults to 'user_profile.json'.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str = "user_profile.json") -> "UserProfile":
        """
        Loads the user profile from a JSON file.

        Args:
            path: Path to the JSON file. Defaults to 'user_profile.json'.

        Returns:
            A UserProfile instance loaded from file, or a new empty one.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.model_validate_json(f.read())
        except FileNotFoundError:
            return cls()
