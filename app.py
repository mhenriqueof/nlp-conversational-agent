"""
Gradio UI for the Conversational Agent.
Entry point for HuggingFace Spaces deployment.
Supports per-user memory and profile via username login.
"""

import gradio as gr
from agent.retrieval.rag import RAGPipeline
from agent.profile.schema import UserProfile


def login(username: str) -> tuple:
    """
    Validates the username and initializes the user session.

    Args:
        username: The username entered by the user.

    Returns:
        A tuple of (login visibility, chat visibility, pipeline, profile text).
    """
    if not username.strip():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            "No profile data yet.",
        )

    pipeline = RAGPipeline(username=username.strip().lower())
    profile = UserProfile.load(username=username.strip().lower())
    profile_text = profile.to_prompt_string() or "**No profile data yet.**"

    return (gr.update(visible=False), gr.update(visible=True), pipeline, profile_text)


def respond(message: str, history: list, pipeline: RAGPipeline, username: str) -> tuple:
    """
    Handles the full response cycle including chat history and profile update.

    Args:
        message: The current message from the user.
        history: The current conversation history.
        pipeline: the RAGPipeline instance scoped to the current user.
        username: The current user's username.

    Returns:
        A tuple of (empty string, update history, updated profile text).
    """
    message = message.strip()

    if not message or pipeline is None:
        return "", history, "No profile data yet."

    response = pipeline.run(message)
    profile = UserProfile.load(username=username.strip().lower())
    profile_text = profile.to_prompt_string() or "**No profile data yet.**"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return "", history, profile_text


with gr.Blocks(
    title="Conversational Agent",
    css="""
        .generating { display: none !important; }
        .eta-bar { display: none !important; }
        .progress-text { display: none !important; }
        .eta-text { display: none !important; }
        span.time { display: none !important; }
        .meta-text { display: none !important; }
        .meta-text-center { display: none !important; }
    """,
) as demo:
    # State
    pipeline_state = gr.State(None)

    # Header
    gr.Markdown("# Conversational Agent")
    gr.Markdown(
        "A conversational agent with persistent memory and dynamic user profile."
    )

    # Login screen
    with gr.Column(visible=True) as login_screen:
        gr.Markdown("### Enter your username to chat")
        username_input = gr.Textbox(
            placeholder="e.g. batman2026", label="Username", lines=1
        )
        login_btn = gr.Button("Enter", variant="primary")
        gr.Markdown("*Your username identifies your personal memory and profile.*")

    # Chat screen
    with gr.Column(visible=False) as chat_screen:
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
                profile_box = gr.Markdown(
                    value="**No profile data yet.**",
                )

    # Login event
    login_btn.click(
        fn=login,
        inputs=[username_input],
        outputs=[login_screen, chat_screen, pipeline_state, profile_box],
    )
    username_input.submit(
        fn=login,
        inputs=[username_input],
        outputs=[login_screen, chat_screen, pipeline_state, profile_box],
    )

    # Chat events
    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot, pipeline_state, username_input],
        outputs=[msg, chatbot, profile_box],
    )
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, pipeline_state, username_input],
        outputs=[msg, chatbot, profile_box],
    )

if __name__ == "__main__":
    demo.launch()
