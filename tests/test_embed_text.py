from agent.memory.embedder import embed_text

v = embed_text('Salut, World!')
print(f"Vector size: {len(v)}\n\
        First 5 values: {v[:5]}")
