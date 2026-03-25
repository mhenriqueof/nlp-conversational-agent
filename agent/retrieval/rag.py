"""
Responsible for the Retrieval-Augmented Generation pipeline.
"""

from agent.memory.embedder import embed_text
from agent.memory.store import MemoryStore
from agent.llm.client import LLMClient


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline for memory-aware response generation.
    Embeds user input, retrieves similar past episodes and generates a context-enriched response.

    Attributes:
        _memory: Private MemoryStore instance for saving and retrieving episodes.
        _llm: Private LLMClient instance for generating responses.
    """

    def __init__(self):
        """
        Initializes the RAGPipeline with a MemoryStore and LLMClient.
        """
        self._memory = MemoryStore()
        self._llm = LLMClient()

    def run(self, user_message: str) -> str:
        """
        Runs the full RAG pipeline for a given user message.

        Steps:
            1. Embed the user message.
            2. Retrieve similar past episodes from memory.
            3. Generate a response using the LLM with memory context.
            4. Save the new episode to memory.

        Args:
            user_message: The current message from the user.

        Returns:
            The generated response as a string.
        """
        # 1. Embed user message
        embedding = embed_text(user_message)

        # 2. Retrieve similar past episodes
        similar_episodes = self._memory.retrieve_similar_episodes(embedding)

        # 3. Generate rersponse with memory context
        response = self._llm.generate_response(
            user_message=user_message, memory_context=similar_episodes
        )

        # 4. Save episode to memory
        self._memory.save_episode(
            user_message=user_message, agent_response=response, embedding=embedding
        )

        return response
