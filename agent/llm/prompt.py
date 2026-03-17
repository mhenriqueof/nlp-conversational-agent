"""
Responsible for building structured prompts for the LLM.
"""


def build_prompt(
    user_message: str,
    memory_context: list[str] = [],
) -> str:
    """
    Builds a structured prompt with optional memory context.

    Args:
        user_message: The current message from the user.
        memory_context: List of relevant past interactions retrieved from memory.

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

    if memory_context:
        memory_block = "\n".join(f"- {m}" for m in memory_context)
        system_prompt += (
            f"\n\n[Relevant memories from past interactions]:\n{memory_block}\n"
        )

    return f"{system_prompt}\n\nUser: {user_message}"
