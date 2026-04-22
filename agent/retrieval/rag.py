"""
Responsible for the Retrieval-Augmented Generation pipeline.
Connects the embedding, memory, profile and LLM layers into a single memory-aware and profile-aware
conversational flow.
Each pipeline instance is scoped to a specific user.
"""

from agent.memory.embedder import embed_text
from agent.memory.store import MemoryStore
from agent.llm.client import LLMClient
from agent.profile.schema import UserProfile
from agent.profile.updater import ProfileUpdater


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline for memory-aware and profile-aware response generation.
    Each instance is scoped to a specific user.

    Attributes:
        _username: The username identifying the current user.
        _memory: Private MemoryStore instance scoped to the user.
        _llm: Private LLMClient instance for generating responses.
        _updater: Private ProfileUpdater instance for updating the user profile.
    """

    def __init__(self, username: str):
        """
        Initializes the RAGPipeline for a specific user.

        Args:
            username: The username used to scope memory and profile.
        """
        self._username = username
        self._memory = MemoryStore(username=username)
        self._llm = LLMClient()
        self._updater = ProfileUpdater()

    def run(self, user_message: str) -> str:
        """
        Runs the full RAG pipeline for a given user message.

        Steps:
            1. Load user profile from disk.
            2. Embed the user message.
            3. Retrieve similar past episodes from memory.
            4. Generate a profile-aware and memory-aware response.
            5. Save the new episode to memory.
            6. Update the user profile with new information.

        Args:
            user_message: The current message from the user.

        Returns:
            The generated response as a string.
        """
        # 1. Load user profile
        profile = UserProfile.load(username=self._username)

        # 2. Embed user message
        embedding = embed_text(user_message)

        # 3. Retrieve similar past episodes
        similar_episodes = self._memory.retrieve_similar_episodes(embedding)

        # 4. Generate response with profile and memory context
        response = self._llm.generate_response(
            user_message=user_message, memory_context=similar_episodes, profile=profile
        )

        # 5. Save episode to memory
        self._memory.save_episode(
            user_message=user_message, agent_response=response, embedding=embedding
        )

        # 6. Update profile with new information
        self._updater.extract_and_update(
            user_message=user_message, profile=profile, username=self._username
        )

        return response
