"""
Tests for Embedding Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Skip tests if sentence-transformers not installed
pytest.importorskip("sentence_transformers")


from src.embeddings.embedder import (
    EmbeddingGenerator,
    EmbeddingConfig,
    CachedEmbeddingGenerator
)
from src.data.chunking import Chunk


class TestEmbeddingConfig:
    """Test EmbeddingConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EmbeddingConfig()
        
        assert "multilingual" in config.model_name.lower()
        assert config.batch_size == 32
        assert config.normalize is True


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""
    
    @pytest.fixture(scope="class")
    def embedder(self):
        """Create embedder instance (cached for performance)."""
        config = EmbeddingConfig()
        return EmbeddingGenerator(config)
    
    def test_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder is not None
        assert embedder.dimension > 0
    
    def test_embed_text(self, embedder):
        """Test single text embedding."""
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        
        assert embedding is not None
        assert len(embedding) == embedder.dimension
        assert isinstance(embedding, np.ndarray)
    
    def test_embed_texts(self, embedder):
        """Test batch text embedding."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]
        embeddings = embedder.embed_texts(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == embedder.dimension for e in embeddings)
    
    def test_embed_chunks(self, embedder):
        """Test embedding chunks."""
        chunks = [
            Chunk(content="Chunk 1", chunk_id="c1", doc_id="d1", chunk_index=0),
            Chunk(content="Chunk 2", chunk_id="c2", doc_id="d1", chunk_index=1)
        ]
        
        embeddings = embedder.embed_chunks(chunks)
        
        assert len(embeddings) == 2
    
    def test_embedding_normalization(self, embedder):
        """Test that embeddings are normalized."""
        text = "Test sentence for normalization."
        embedding = embedder.embed_text(text)
        
        # L2 norm should be approximately 1 if normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01
    
    def test_similarity(self, embedder):
        """Test similarity calculation."""
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "The weather is sunny today."
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)
        
        sim_12 = embedder.similarity(emb1, emb2)
        sim_13 = embedder.similarity(emb1, emb3)
        
        # Similar sentences should have higher similarity
        assert sim_12 > sim_13
    
    def test_empty_text(self, embedder):
        """Test embedding empty text."""
        embedding = embedder.embed_text("")
        
        # Should still return an embedding
        assert embedding is not None
        assert len(embedding) == embedder.dimension
    
    def test_multilingual(self, embedder):
        """Test multilingual embeddings."""
        text_en = "Hello world"
        text_fr = "Bonjour le monde"
        text_ar = "مرحبا بالعالم"
        
        emb_en = embedder.embed_text(text_en)
        emb_fr = embedder.embed_text(text_fr)
        emb_ar = embedder.embed_text(text_ar)
        
        # All should produce valid embeddings
        assert all(len(e) == embedder.dimension for e in [emb_en, emb_fr, emb_ar])
        
        # Similar meanings should have higher similarity
        sim_en_fr = embedder.similarity(emb_en, emb_fr)
        assert sim_en_fr > 0.5  # Should be reasonably similar


class TestCachedEmbeddingGenerator:
    """Test CachedEmbeddingGenerator class."""
    
    @pytest.fixture
    def cached_embedder(self):
        """Create cached embedder instance."""
        config = EmbeddingConfig()
        return CachedEmbeddingGenerator(config, cache_size=100)
    
    def test_caching(self, cached_embedder):
        """Test that caching works."""
        text = "Test sentence for caching."
        
        # First call
        emb1 = cached_embedder.embed_text(text)
        
        # Second call (should be cached)
        emb2 = cached_embedder.embed_text(text)
        
        # Should return same embedding
        assert np.allclose(emb1, emb2)
    
    def test_cache_stats(self, cached_embedder):
        """Test cache statistics."""
        texts = ["Text 1", "Text 2", "Text 1"]  # Text 1 repeated
        
        for text in texts:
            cached_embedder.embed_text(text)
        
        stats = cached_embedder.get_cache_stats()
        
        assert stats["size"] == 2  # Only 2 unique texts
        assert stats["hits"] >= 1  # At least one cache hit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
