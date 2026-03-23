"""
Gradio UI for Conversational Agent.
Entry point for Hugging Face Spaces deployment.
"""

import gradio as gr
from agent.retrieval.rag import RAGPipeline

pipeline = RAGPipeline()


def chat(message: str, history: list) -> str:
    """
    Processes a user message and returns the agent's response.

    Args:
        message: The current message from the user.
        history: The conversation history maintained by Gradio.

    Returns:
        The generated response as a string.
    """
    response = pipeline.run(message)
    return response


demo = gr.ChatInterface(
    fn=chat,
    title="Conversational Agent",
    description="A conversational agent with persistent memory. It remembers past interactions.",
    examples=[
        "Hello!",
        "What's your name?",
        "What do you know about me so far?",
    ],
)

if __name__ == "__main__":
    demo.launch()
