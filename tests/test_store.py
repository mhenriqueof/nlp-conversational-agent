from agent.memory.embedder import embed_text
from agent.memory.store import save_episode, retrieve_similar_episodes

embedding = embed_text('I love ESO')
save_episode('I love ESO', 'Great, ESO is great!', embedding)

results = retrieve_similar_episodes(embed_text('What food do I like?'))
print('Retrieved episodes:')
for r in results: print(f'  - {r}')
