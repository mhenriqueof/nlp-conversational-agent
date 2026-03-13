from agent.memory.embedder import embed_text
from agent.memory.store import MemoryStore

memory = MemoryStore()
embedding = embed_text("I love ESO")
memory.save_episode("I love ESO", "Great, ESO is great!", embedding)

results = memory.retrieve_similar_episodes(embed_text("What food do I like?"))
print("Retrieved episodes:")
for r in results:
    print(f"  - {r}")
