from agent.retrieval.rag import RAGPipeline

pipeline = RAGPipeline(username="batman2026")
response = pipeline.run("Hi! My name is Bruce and I love martial arts.")
print(f"Agent: {response}")
