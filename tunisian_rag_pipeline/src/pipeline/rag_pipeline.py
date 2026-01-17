"""
RAG Pipeline Module
===================
Main pipeline orchestrating the complete RAG workflow.
User Query → Intent → Retrieval → Evidence Packing → LLM → Answer
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from loguru import logger

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.ingestion import DataIngestion, Document
from src.data.preprocessing import TextPreprocessor
from src.data.chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy
from src.embeddings.embedder import EmbeddingGenerator, EmbeddingConfig
from src.retrieval.vector_store import VectorStore, VectorStoreConfig
from src.retrieval.retriever import Retriever, RetrieverConfig, RetrievalResult
from src.llm.generator import LLMGenerator, SimpleLLMGenerator, LLMConfig, LLMProvider
from src.llm.prompts import format_context_for_prompt, SYSTEM_PROMPT_RAG
from src.pipeline.intent import IntentClassifier, IntentResult, QueryIntent


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    # Paths
    data_dir: str = "../tunisian_heritage_data"
    vector_db_dir: str = "./vector_db"
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "semantic"
    
    # Embedding
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 32
    
    # Retrieval
    top_k: int = 5
    min_score: float = 0.1  # Lower threshold for better recall
    use_rerank: bool = False
    use_mmr: bool = False  # Disable MMR by default for speed
    
    # LLM - Default to LM Studio
    llm_provider: str = "lmstudio"
    llm_model: str = "saka-14b-i1"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # General
    use_intent_classification: bool = True
    max_context_length: int = 3000


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    intent: Optional[str]
    language: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "intent": self.intent,
            "language": self.language,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """Pretty print the response."""
        lines = [
            "=" * 60,
            "ANSWER:",
            self.answer,
            "",
            "SOURCES:",
        ]
        
        for i, source in enumerate(self.sources, 1):
            lines.append(f"  [{i}] {source.get('source_info', 'Unknown')} (score: {source.get('score', 0):.3f})")
        
        lines.extend([
            "",
            f"Confidence: {self.confidence:.2f}",
            f"Intent: {self.intent}",
            f"Language: {self.language}",
            f"Processing time: {self.processing_time:.2f}s",
            "=" * 60
        ])
        
        return "\n".join(lines)


class RAGPipeline:
    """
    Complete RAG Pipeline for Tunisian Heritage Q&A.
    
    Pipeline flow:
    1. Intent Classification - Understand the query
    2. Query Embedding - Convert query to vector
    3. Retrieval - Find relevant chunks
    4. Evidence Packing - Format context for LLM
    5. Generation - Generate answer with LLM
    6. Response Formatting - Build final response
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or RAGConfig()
        
        # Components (lazy loaded)
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._llm = None
        self._intent_classifier = None
        
        self._initialized = False
        
        logger.info("RAGPipeline created")
    
    def _init_embedder(self) -> EmbeddingGenerator:
        """Initialize embedding generator."""
        if self._embedder is None:
            config = EmbeddingConfig(
                model_name=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size
            )
            self._embedder = EmbeddingGenerator(config)
        return self._embedder
    
    def _init_vector_store(self) -> VectorStore:
        """Initialize vector store."""
        if self._vector_store is None:
            config = VectorStoreConfig(
                persist_directory=self.config.vector_db_dir,
                collection_name="tunisian_heritage"
            )
            self._vector_store = VectorStore(config)
        return self._vector_store
    
    def _init_retriever(self) -> Retriever:
        """Initialize retriever."""
        if self._retriever is None:
            embedder = self._init_embedder()
            vector_store = self._init_vector_store()
            
            config = RetrieverConfig(
                top_k=self.config.top_k,
                min_score=self.config.min_score,
                use_rerank=self.config.use_rerank,
                use_mmr=getattr(self.config, 'use_mmr', False)
            )
            self._retriever = Retriever(vector_store, embedder, config)
        return self._retriever
    
    def _init_llm(self) -> Any:
        """Initialize LLM generator."""
        if self._llm is None:
            # Try to use full LLM generator
            try:
                config = LLMConfig(
                    provider=LLMProvider(self.config.llm_provider),
                    model_name=self.config.llm_model,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    lmstudio_base_url=getattr(self.config, 'lmstudio_base_url', 'http://localhost:1234/v1')
                )
                self._llm = LLMGenerator(config)
                logger.info(f"LLM initialized: {self.config.llm_provider} / {self.config.llm_model}")
            except Exception as e:
                logger.warning(f"Could not initialize full LLM: {e}")
                logger.info("Falling back to SimpleLLMGenerator")
                self._llm = SimpleLLMGenerator()
        return self._llm
    
    def _init_intent_classifier(self) -> IntentClassifier:
        """Initialize intent classifier."""
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifier()
        return self._intent_classifier
    
    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info("Initializing RAG pipeline components...")
        
        self._init_embedder()
        self._init_vector_store()
        self._init_retriever()
        self._init_llm()
        
        if self.config.use_intent_classification:
            self._init_intent_classifier()
        
        self._initialized = True
        logger.info("RAG pipeline initialized")
    
    def query(
        self,
        question: str,
        language: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            language: Optional language filter
            filters: Additional metadata filters
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and sources
        """
        import time
        start_time = time.time()
        
        self.initialize()
        
        # Step 1: Intent Classification
        intent_result = None
        if self.config.use_intent_classification:
            classifier = self._init_intent_classifier()
            intent_result = classifier.classify(question)
            logger.info(f"Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})")
            
            # Use detected language if not specified
            if language is None:
                language = intent_result.language
        
        # Step 2: Retrieve relevant documents
        retriever = self._init_retriever()
        
        # Don't filter by language since chunks may not have language metadata
        results = retriever.retrieve(
            query=question,
            top_k=top_k or self.config.top_k,
            filters=filters,
            language=None  # Don't filter by language
        )
        
        if not results:
            return RAGResponse(
                answer="I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
                sources=[],
                confidence=0.0,
                intent=intent_result.intent.value if intent_result else None,
                language=language or "en",
                processing_time=time.time() - start_time
            )
        
        # Step 3: Pack evidence into context
        context = format_context_for_prompt(
            [{"content": r.content, "source_info": r.source_info} for r in results],
            max_length=self.config.max_context_length
        )
        
        # Step 4: Generate answer
        llm = self._init_llm()
        
        try:
            if isinstance(llm, SimpleLLMGenerator):
                answer = llm.generate(question, context)
            else:
                result = llm.generate_with_context(question, context)
                answer = result.answer
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            answer = self._fallback_answer(question, results)
        
        # Step 5: Calculate confidence
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        confidence = min(avg_score * 1.2, 1.0)  # Slight boost, cap at 1.0
        
        # Step 6: Build response
        processing_time = time.time() - start_time
        
        sources = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "score": r.score,
                "source_info": r.source_info,
                "metadata": r.metadata
            }
            for r in results
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            intent=intent_result.intent.value if intent_result else None,
            language=language or "en",
            processing_time=processing_time,
            metadata={
                "num_sources": len(results),
                "avg_retrieval_score": avg_score
            }
        )
    
    def _fallback_answer(self, question: str, results: List[RetrievalResult]) -> str:
        """Generate a fallback answer when LLM fails."""
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Use top result as basis
        top_result = results[0]
        
        return f"""Based on the available information:

{top_result.content[:500]}...

Source: {top_result.source_info}

Note: This is a direct excerpt from the knowledge base. For a more comprehensive answer, ensure the LLM component is properly configured."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        self.initialize()
        
        vector_store = self._init_vector_store()
        stats = vector_store.get_collection_stats()
        
        return {
            "vector_db": stats,
            "config": {
                "embedding_model": self.config.embedding_model,
                "llm_model": self.config.llm_model,
                "top_k": self.config.top_k
            },
            "initialized": self._initialized
        }


def create_pipeline(
    data_dir: str = "../tunisian_heritage_data",
    vector_db_dir: str = "./vector_db",
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        data_dir: Path to data directory
        vector_db_dir: Path to vector database
        **kwargs: Additional config options
        
    Returns:
        RAGPipeline instance
    """
    config = RAGConfig(
        data_dir=data_dir,
        vector_db_dir=vector_db_dir,
        **kwargs
    )
    return RAGPipeline(config)


if __name__ == "__main__":
    # Quick test
    print("RAG Pipeline Module")
    print("=" * 60)
    
    print("\nTo use the pipeline:")
    print("  from src.pipeline.rag_pipeline import create_pipeline")
    print("  pipeline = create_pipeline()")
    print("  response = pipeline.query('What caused the Tunisian revolution?')")
    print("  print(response)")
