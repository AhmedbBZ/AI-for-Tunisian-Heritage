"""Quick test of vector store retrieval."""
from src.retrieval.vector_store import VectorStore, VectorStoreConfig
from src.embeddings.embedder import EmbeddingGenerator, EmbeddingConfig

# Initialize
config = VectorStoreConfig(persist_directory='./vector_db', collection_name='tunisian_heritage')
store = VectorStore(config)
embedder = EmbeddingGenerator(EmbeddingConfig())

# Check stats
stats = store.get_collection_stats()
print(f'Collection stats: {stats}')

# Test a simple search
query = 'Tunisia history'
query_emb = embedder.embed_text(query)
results = store.search(query_emb, top_k=3)
print(f'\nResults for "{query}":')
for r in results:
    print(f'  Score: {r.score:.3f} - {r.content[:100]}...')
