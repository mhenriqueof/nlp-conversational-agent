"""
Responsible for building structured prompts for the LLM.
"""

from agent.profile.schema import UserProfile


def build_prompt(
    user_message: str,
    memory_context: list[str] = [],
    profile: UserProfile | None = None,
) -> str:
    """
    Builds a structured prompt with optional memory context and user profile.

    Args:
        user_message: The current message from the user.
        memory_context: List of relevant past interactions retrieved from memory.
        profile: The current UserProfile instance for personalized responses.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    system_prompt = (
        "You are a helpful, honest and friendly conversational agent. "
        "You have access to memory of past interactions with the user. "
        "Use that memory naturally and sparingly - only when clearly relevant. "
        "Never force memory into the conversation. "
        "Respond naturally to what the user just said, don't bring up past topics unprompted."
    )

    # Inject user profile if available
    if profile:
        profile_string = profile.to_prompt_string()
        if profile_string:
            system_prompt += f"\n\n{profile_string}"

    # Inject memory context if available
    if memory_context:
        memory_block = "\n".join(f"- {m}" for m in memory_context)
        system_prompt += (
            f"\n\n[Relevant memories from past interactions]:\n{memory_block}\n"
        )

    return f"{system_prompt}\n\nUser: {user_message}"
