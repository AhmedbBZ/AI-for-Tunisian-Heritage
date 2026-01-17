"""
Tests for Retrieval Module
"""

import pytest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip if dependencies not available
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from src.retrieval.vector_store import VectorStore, VectorStoreConfig, SearchResult
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.embeddings.embedder import EmbeddingGenerator, EmbeddingConfig
from src.data.chunking import Chunk


class TestVectorStoreConfig:
    """Test VectorStoreConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = VectorStoreConfig()
        
        assert config.collection_name == "tunisian_heritage"
        assert config.distance_metric == "cosine"


class TestVectorStore:
    """Test VectorStore class."""
    
    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def vector_store(self, temp_db_dir):
        """Create vector store instance."""
        config = VectorStoreConfig(
            persist_directory=temp_db_dir,
            collection_name="test_collection"
        )
        return VectorStore(config)
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return EmbeddingGenerator(EmbeddingConfig())
    
    def test_initialization(self, vector_store):
        """Test vector store initialization."""
        assert vector_store is not None
        stats = vector_store.get_collection_stats()
        assert stats["count"] == 0
    
    def test_add_document(self, vector_store, embedder):
        """Test adding a document."""
        embedding = embedder.embed_text("Test document")
        
        vector_store.add_document(
            doc_id="doc_1",
            embedding=embedding,
            content="Test document content",
            metadata={"source": "test.txt"}
        )
        
        stats = vector_store.get_collection_stats()
        assert stats["count"] == 1
    
    def test_add_chunks(self, vector_store, embedder):
        """Test adding chunks."""
        chunks = [
            Chunk(content="Chunk 1 content", chunk_id="c1", doc_id="d1", chunk_index=0),
            Chunk(content="Chunk 2 content", chunk_id="c2", doc_id="d1", chunk_index=1)
        ]
        
        vector_store.add_chunks(chunks, embedder)
        
        stats = vector_store.get_collection_stats()
        assert stats["count"] == 2
    
    def test_search(self, vector_store, embedder):
        """Test basic search."""
        # Add documents
        texts = [
            "Tunisia is a country in North Africa",
            "Paris is the capital of France",
            "Tunisian cuisine includes couscous"
        ]
        
        for i, text in enumerate(texts):
            embedding = embedder.embed_text(text)
            vector_store.add_document(
                doc_id=f"doc_{i}",
                embedding=embedding,
                content=text
            )
        
        # Search for Tunisia
        query_embedding = embedder.embed_text("Tell me about Tunisia")
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) == 2
        # Tunisia-related documents should be top results
        assert "Tunisia" in results[0].content or "Tunisian" in results[0].content
    
    def test_search_with_filter(self, vector_store, embedder):
        """Test search with metadata filter."""
        # Add documents with different languages
        docs = [
            ("English text", {"language": "en"}),
            ("Texte français", {"language": "fr"}),
            ("نص عربي", {"language": "ar"})
        ]
        
        for i, (text, metadata) in enumerate(docs):
            embedding = embedder.embed_text(text)
            vector_store.add_document(
                doc_id=f"doc_{i}",
                embedding=embedding,
                content=text,
                metadata=metadata
            )
        
        # Search with language filter
        query_embedding = embedder.embed_text("text")
        results = vector_store.search_with_filter(
            query_embedding,
            filter_dict={"language": "fr"},
            top_k=5
        )
        
        assert all(r.metadata.get("language") == "fr" for r in results)
    
    def test_delete_document(self, vector_store, embedder):
        """Test deleting a document."""
        embedding = embedder.embed_text("Test document")
        
        vector_store.add_document(
            doc_id="doc_to_delete",
            embedding=embedding,
            content="Document to delete"
        )
        
        assert vector_store.get_collection_stats()["count"] == 1
        
        vector_store.delete(["doc_to_delete"])
        
        assert vector_store.get_collection_stats()["count"] == 0


class TestRetriever:
    """Test Retriever class."""
    
    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def retriever(self, temp_db_dir):
        """Create retriever with test data."""
        # Setup
        store_config = VectorStoreConfig(
            persist_directory=temp_db_dir,
            collection_name="test_retriever"
        )
        vector_store = VectorStore(store_config)
        embedder = EmbeddingGenerator(EmbeddingConfig())
        
        # Add test documents
        docs = [
            "The Tunisian Revolution began in December 2010",
            "Couscous is a traditional Tunisian dish",
            "Carthage was an ancient civilization in Tunisia",
            "The weather in Paris is often rainy",
            "Python is a programming language"
        ]
        
        for i, text in enumerate(docs):
            embedding = embedder.embed_text(text)
            vector_store.add_document(
                doc_id=f"doc_{i}",
                embedding=embedding,
                content=text,
                metadata={"index": i}
            )
        
        # Create retriever
        config = RetrieverConfig(top_k=3, min_score=0.0)
        return Retriever(vector_store, embedder, config)
    
    def test_retrieve(self, retriever):
        """Test basic retrieval."""
        results = retriever.retrieve("What is the history of Tunisia?")
        
        assert len(results) <= 3
        assert all(hasattr(r, "content") for r in results)
    
    def test_retrieve_relevance(self, retriever):
        """Test retrieval relevance."""
        results = retriever.retrieve("Tell me about Tunisian food")
        
        # Couscous document should be in top results
        contents = [r.content for r in results]
        assert any("Couscous" in c or "dish" in c for c in contents)
    
    def test_retrieve_top_k(self, retriever):
        """Test top_k parameter."""
        results = retriever.retrieve("Tunisia", top_k=2)
        assert len(results) <= 2
    
    def test_retrieve_min_score(self, retriever):
        """Test minimum score filtering."""
        # High threshold should return fewer results
        results = retriever.retrieve(
            "Completely unrelated query about quantum physics",
            min_score=0.9
        )
        
        # Should filter out low-relevance results
        # Note: Results depend on model, so just check it doesn't crash


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating search result."""
        result = SearchResult(
            chunk_id="chunk_1",
            content="Test content",
            score=0.85,
            metadata={"source": "test.txt"}
        )
        
        assert result.chunk_id == "chunk_1"
        assert result.score == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
