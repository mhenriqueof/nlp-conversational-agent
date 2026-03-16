from agent.retrieval.rag import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.run("Hello! My name is Henrique.")
print(f"Agent: {response}")
