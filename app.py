"""
Gradio UI for the Conversational Agent.
Entry point for HuggingFace Spaces deployment.
"""

import gradio as gr
from agent.retrieval.rag import RAGPipeline
from agent.profile.schema import UserProfile

pipeline = RAGPipeline()


def chat(message: str, history: list) -> tuple[str, str]:
    """
    Processes a user message and returns the agent's response
    and the updated profile string.

    Args:
        message: The current message from the user.
        history: The conversation history maintained by Gradio.

    Returns:
        A tuple of (agent response, updated profile string).
    """
    response = pipeline.run(message)
    profile = UserProfile.load()
    profile_text = profile.to_prompt_string() or "No profile data yet."
    return response, profile_text


with gr.Blocks(title="Conversational Agent") as demo:
    gr.Markdown("# Conversational Agent")
    gr.Markdown(
        "A conversational agent with persistent memory and dynamic user profile."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="Type your message here...",
                label="Your message",
                lines=2,
            )
            send_btn = gr.Button("Send", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Your Profile")
            profile_box = gr.Textbox(
                label="Profile",
                lines=12,
                interactive=False,
                value="No profile data yet.",
            )

    def respond(message: str, history: list) -> tuple:
        """
        Handles the full response cycle including chat history update.

        Args:
            message: The current message from the user.
            history: The current conversation history.

        Returns:
            A tuple of (empty string, updated history, updated profile text).
        """
        if not message.strip():
            return "", history, "No profile data yet."

        response, profile_text = chat(message, history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return "", history, profile_text

    send_btn.click(
        fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot, profile_box]
    )
    msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot, profile_box])

if __name__ == "__main__":
    demo.launch()
