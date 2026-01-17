"""
Retriever Module
================
Implements semantic search with re-ranking and diversity optimization.
Combines vector similarity with metadata filtering.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from loguru import logger

from .vector_store import VectorStore, SearchResult, create_vector_store


@dataclass
class RetrieverConfig:
    """Configuration for retriever."""
    top_k: int = 5
    min_score: float = 0.3
    use_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_mmr: bool = True
    mmr_lambda: float = 0.7  # Balance between relevance and diversity


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with additional metadata."""
    chunk_id: str
    content: str
    score: float
    rerank_score: Optional[float]
    metadata: Dict[str, Any]
    source_info: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
            "source_info": self.source_info
        }


class Retriever:
    """
    Advanced retriever with re-ranking and diversity optimization.
    
    Features:
    - Vector similarity search
    - Metadata filtering
    - Cross-encoder re-ranking (optional)
    - Maximal Marginal Relevance for diversity
    - Multi-language support
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: Any,  # EmbeddingGenerator
        config: Optional[RetrieverConfig] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            config: Retriever configuration
        """
        self.vector_store = vector_store
        self.embedder = embedding_generator
        self.config = config or RetrieverConfig()
        self.reranker = None
        
        logger.info("Initialized Retriever")
    
    def _load_reranker(self) -> None:
        """Load the re-ranking model."""
        if self.reranker is not None or not self.config.use_rerank:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading reranker: {self.config.rerank_model}")
            self.reranker = CrossEncoder(self.config.rerank_model)
            logger.info("Reranker loaded")
        
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            self.config.use_rerank = False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        doc_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (overrides config)
            filters: Additional metadata filters
            language: Filter by language
            doc_type: Filter by document type
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self.config.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Initial retrieval (get more if reranking)
        retrieve_k = top_k * 3 if self.config.use_rerank else top_k * 2
        
        # Build filters
        where = None
        if filters or language or doc_type:
            conditions = []
            
            if language:
                conditions.append({"language": language})
            
            if doc_type:
                conditions.append({"file_type": doc_type})
            
            if filters:
                conditions.append(filters)
            
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}
        
        # Vector search
        search_results = self.vector_store.search(
            query_embedding,
            top_k=retrieve_k,
            where=where
        )
        
        # Filter by minimum score
        search_results = [r for r in search_results if r.score >= self.config.min_score]
        
        if not search_results:
            logger.warning(f"No results found for query: {query[:50]}...")
            return []
        
        # Apply MMR for diversity
        if self.config.use_mmr and len(search_results) > top_k:
            search_results = self._apply_mmr(
                query_embedding,
                search_results,
                top_k=top_k * 2 if self.config.use_rerank else top_k
            )
        
        # Re-rank if enabled
        if self.config.use_rerank:
            self._load_reranker()
            if self.reranker:
                search_results = self._rerank(query, search_results, top_k)
        
        # Convert to RetrievalResult
        results = []
        for sr in search_results[:top_k]:
            source_info = self._build_source_info(sr.metadata)
            
            results.append(RetrievalResult(
                chunk_id=sr.chunk_id,
                content=sr.content,
                score=sr.score,
                rerank_score=getattr(sr, 'rerank_score', None),
                metadata=sr.metadata,
                source_info=source_info
            ))
        
        logger.info(f"Retrieved {len(results)} results for query: {query[:30]}...")
        return results
    
    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        MMR balances relevance to query with diversity among results.
        """
        if len(results) <= top_k:
            return results
        
        # Get embeddings for results (approximate from content)
        result_embeddings = []
        for r in results:
            # Use the embedding generator to get embeddings
            emb = self.embedder.embed_text(r.content)
            result_embeddings.append(emb)
        
        result_embeddings = np.array(result_embeddings)
        
        # Ensure query is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        # Calculate query similarities (already have scores)
        query_sims = np.array([r.score for r in results])
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_sims[idx]
                
                # Max similarity to already selected
                if selected_indices:
                    selected_embs = result_embeddings[selected_indices]
                    sims = np.dot(selected_embs, result_embeddings[idx])
                    max_sim = np.max(sims)
                else:
                    max_sim = 0
                
                # MMR score
                mmr = self.config.mmr_lambda * relevance - (1 - self.config.mmr_lambda) * max_sim
                mmr_scores.append((idx, mmr))
            
            # Select highest MMR
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [results[i] for i in selected_indices]
    
    def _rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Re-rank results using cross-encoder."""
        if not self.reranker or not results:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.content) for r in results]
        
        # Get re-ranking scores
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for i, r in enumerate(results):
            r.rerank_score = float(scores[i])
        
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return results[:top_k]
    
    def _build_source_info(self, metadata: Dict[str, Any]) -> str:
        """Build human-readable source information."""
        parts = []
        
        if metadata.get("source_file"):
            parts.append(f"Source: {metadata['source_file']}")
        
        if metadata.get("doc_id"):
            parts.append(f"Document: {metadata['doc_id']}")
        
        if metadata.get("chunk_index") is not None:
            parts.append(f"Chunk: {metadata['chunk_index']}")
        
        if metadata.get("language"):
            parts.append(f"Language: {metadata['language']}")
        
        if metadata.get("type"):
            parts.append(f"Type: {metadata['type']}")
        
        return " | ".join(parts) if parts else "Unknown source"
    
    def retrieve_by_language(
        self,
        query: str,
        language: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents in a specific language.
        
        Args:
            query: Search query
            language: Language code (ar, en, fr)
            top_k: Number of results
            
        Returns:
            List of results
        """
        return self.retrieve(query, top_k=top_k, language=language)
    
    def retrieve_multilingual(
        self,
        query: str,
        languages: List[str] = ["en", "fr", "ar"],
        top_k_per_lang: int = 2
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve documents from multiple languages.
        
        Args:
            query: Search query
            languages: List of language codes
            top_k_per_lang: Results per language
            
        Returns:
            Dictionary mapping language to results
        """
        results = {}
        
        for lang in languages:
            lang_results = self.retrieve(query, top_k=top_k_per_lang, language=lang)
            if lang_results:
                results[lang] = lang_results
        
        return results
    
    def get_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to find
            
        Returns:
            List of similar chunks
        """
        # Get the chunk
        chunk = self.vector_store.get_document(chunk_id)
        
        if not chunk:
            logger.warning(f"Chunk not found: {chunk_id}")
            return []
        
        # Use chunk content as query
        return self.retrieve(chunk["content"], top_k=top_k + 1)[1:]  # Exclude self


def create_retriever(
    vector_store: VectorStore,
    embedding_generator: Any,
    **kwargs
) -> Retriever:
    """
    Factory function to create a retriever.
    
    Args:
        vector_store: Vector store instance
        embedding_generator: Embedding generator
        **kwargs: Config options
        
    Returns:
        Retriever instance
    """
    config = RetrieverConfig(**kwargs)
    return Retriever(vector_store, embedding_generator, config)


if __name__ == "__main__":
    print("Retriever module loaded.")
    print("Use create_retriever() to create a retriever instance.")
    print("\nExample:")
    print("  retriever = create_retriever(vector_store, embedder)")
    print("  results = retriever.retrieve('What caused the Tunisian revolution?')")
