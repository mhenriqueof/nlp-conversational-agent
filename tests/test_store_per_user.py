from agent.memory.store import MemoryStore
from agent.memory.embedder import embed_text

store = MemoryStore(username="batman2026")
embedding = embed_text("I love chess")
store.save_episode("I love chess", "Great, chess is a wonderful game!", embedding)

results = store.retrieve_similar_episodes(embed_text("What games do I like?"))
print("Retrieved episodes:")
for r in results:
    print(f"  - {r}")
