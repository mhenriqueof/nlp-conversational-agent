"""
Runs a simple command line conversation loop using the RAG pipeline.
"""

from agent.retrieval.rag import RAGPipeline


def main():
    """
    Runs the CLI conversation loop.
    Initializes the RAG pipeline and starts an interactive session that continues until the
    user types 'exit' or 'quit'.
    """
    print("-" * 50)
    print(" Conversational Agent - CLI")
    print("  Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    pipeline = RAGPipeline()

    while True:
        user_input = input("\n" "You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("\n" "Agent: Goodbye!")
            break

        print("\n" "Agent: Thinking...", end="\r")
        response = pipeline.run(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
