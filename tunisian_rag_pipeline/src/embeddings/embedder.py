"""
Embedding Generation Module
===========================
Generates vector embeddings for text chunks using sentence transformers.
Supports multilingual models for Arabic, French, and English.
"""

import os
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import numpy as np

from loguru import logger

# Lazy imports for optional dependencies
torch = None
SentenceTransformer = None


def _import_torch():
    """Lazy import torch."""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


def _import_sentence_transformers():
    """Lazy import sentence transformers."""
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST
    return SentenceTransformer


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    device: str = "auto"  # auto, cuda, cpu
    show_progress: bool = True


class EmbeddingGenerator:
    """
    Generates embeddings using sentence transformers.
    
    Supports:
    - Multilingual models (Arabic, French, English)
    - GPU acceleration
    - Batch processing
    - Embedding normalization
    """
    
    # Recommended models by use case
    RECOMMENDED_MODELS = {
        "multilingual_fast": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual_quality": "intfloat/multilingual-e5-large",
        "english_fast": "sentence-transformers/all-MiniLM-L6-v2",
        "english_quality": "sentence-transformers/all-mpnet-base-v2",
        "arabic": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    }
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding generator.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = None
        self._embedding_dim = None
        
        logger.info(f"Initialized EmbeddingGenerator with model: {self.config.model_name}")
    
    def _setup_device(self) -> str:
        """Setup and return the compute device."""
        torch = _import_torch()
        
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("Using CPU (GPU not available)")
        else:
            device = self.config.device
        
        return device
    
    def load_model(self) -> None:
        """Load the embedding model."""
        if self.model is not None:
            return
        
        ST = _import_sentence_transformers()
        
        self.device = self._setup_device()
        
        logger.info(f"Loading embedding model: {self.config.model_name}")
        
        try:
            self.model = ST(
                self.config.model_name,
                device=self.device
            )
            
            # Set max sequence length
            self.model.max_seq_length = self.config.max_seq_length
            
            # Get embedding dimension
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            self.load_model()
        return self._embedding_dim
    
    @property
    def dimension(self) -> int:
        """Alias for embedding_dim."""
        return self.embedding_dim
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: Optional[bool] = None
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings (shape: [num_texts, embedding_dim])
        """
        if not texts:
            return np.array([])
        
        self.load_model()
        
        show_progress = show_progress if show_progress is not None else self.config.show_progress
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_chunks(
        self,
        chunks: List[Any],  # List of Chunk objects
        text_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunk objects.
        
        Args:
            chunks: List of chunk objects (with 'content' attribute)
            text_field: Name of the text field
            
        Returns:
            List of dicts with chunk data and embeddings
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = []
        for chunk in chunks:
            if hasattr(chunk, text_field):
                texts.append(getattr(chunk, text_field))
            elif isinstance(chunk, dict) and text_field in chunk:
                texts.append(chunk[text_field])
            else:
                raise ValueError(f"Chunk missing '{text_field}' field")
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Combine chunks with embeddings
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            if hasattr(chunk, 'to_dict'):
                chunk_data = chunk.to_dict()
            elif isinstance(chunk, dict):
                chunk_data = chunk.copy()
            else:
                chunk_data = {"content": str(chunk)}
            
            chunk_data["embedding"] = embedding.tolist()
            results.append(chunk_data)
        
        return results
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Flatten if needed
        e1 = np.asarray(embedding1).flatten()
        e2 = np.asarray(embedding2).flatten()
        
        # Cosine similarity
        if self.config.normalize_embeddings:
            # Already normalized, just dot product
            return float(np.dot(e1, e2))
        else:
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(e1, e2) / (norm1 * norm2))
    
    def rank_by_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Rank documents by similarity to query.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples
        """
        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity (assuming normalized embeddings)
        if self.config.normalize_embeddings:
            scores = np.dot(doc_embeddings, query_embedding.T).flatten()
        else:
            # Manual cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1)
            scores = np.dot(doc_embeddings, query_embedding.T).flatten()
            scores = scores / (doc_norms * query_norm + 1e-8)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        self.load_model()
        
        return {
            "model_name": self.config.model_name,
            "embedding_dim": self._embedding_dim,
            "max_seq_length": self.config.max_seq_length,
            "device": str(self.device),
            "normalize_embeddings": self.config.normalize_embeddings
        }


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator with caching support.
    Caches embeddings to avoid recomputation.
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize with caching.
        
        Args:
            config: Embedding configuration
            cache_dir: Directory for cache storage
        """
        super().__init__(config)
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, np.ndarray] = {}
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding with caching.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            
            # Check memory cache
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
        
        # Generate embedding
        embedding = super().embed_text(text)
        
        if use_cache:
            self.memory_cache[cache_key] = embedding
        
        return embedding
    
    def clear_cache(self) -> None:
        """Clear the memory cache."""
        self.memory_cache.clear()
        logger.info("Cleared embedding cache")


def create_embedder(
    model_name: Optional[str] = None,
    use_cache: bool = False,
    **kwargs
) -> Union[EmbeddingGenerator, CachedEmbeddingGenerator]:
    """
    Factory function to create an embedding generator.
    
    Args:
        model_name: Model name or preset key
        use_cache: Whether to use caching
        **kwargs: Additional config options
        
    Returns:
        Embedding generator instance
    """
    # Check if model_name is a preset
    if model_name in EmbeddingGenerator.RECOMMENDED_MODELS:
        model_name = EmbeddingGenerator.RECOMMENDED_MODELS[model_name]
    
    config = EmbeddingConfig(
        model_name=model_name or EmbeddingConfig.model_name,
        **kwargs
    )
    
    if use_cache:
        return CachedEmbeddingGenerator(config)
    else:
        return EmbeddingGenerator(config)


if __name__ == "__main__":
    # Test the module
    print("Testing EmbeddingGenerator:\n")
    
    # Test texts in different languages
    test_texts = [
        "The Tunisian Revolution began in December 2010.",
        "La révolution tunisienne a commencé en décembre 2010.",
        "بدأت الثورة التونسية في ديسمبر 2010.",
        "Habib Bourguiba was the first president of Tunisia.",
        "Carthage was an ancient city in Tunisia."
    ]
    
    try:
        # Create embedder
        embedder = create_embedder()
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = embedder.embed_texts(test_texts)
        
        print(f"\nEmbedding shape: {embeddings.shape}")
        print(f"Model info: {embedder.get_model_info()}")
        
        # Test similarity
        print("\nSimilarity test:")
        query = "When did the revolution in Tunisia start?"
        query_embedding = embedder.embed_text(query)
        
        similarities = embedder.similarity(query_embedding, embeddings, top_k=3)
        
        print(f"Query: {query}")
        print("\nTop matches:")
        for idx, score in similarities:
            print(f"  [{score:.4f}] {test_texts[idx]}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure sentence-transformers and torch are installed:")
        print("  pip install sentence-transformers torch")
