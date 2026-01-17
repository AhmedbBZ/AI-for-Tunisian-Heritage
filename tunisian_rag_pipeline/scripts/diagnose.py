#!/usr/bin/env python3
"""
Diagnostics Script
==================
Tools for diagnosing, debugging, and monitoring the RAG pipeline.

Features:
- Vector database health checks
- Retrieval quality analysis
- LLM response evaluation
- Performance profiling
- Logging and metrics

Usage:
    python diagnose.py health          # Run health checks
    python diagnose.py test-retrieval  # Test retrieval quality
    python diagnose.py profile         # Profile performance
    python diagnose.py logs            # View recent logs
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # "ok", "warning", "error"
    message: str
    details: Dict[str, Any] = None
    
    def __str__(self):
        emoji = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(self.status, "❓")
        return f"{emoji} {self.component}: {self.message}"


def check_dependencies() -> HealthCheckResult:
    """Check if all required dependencies are installed."""
    missing = []
    warnings = []
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'chromadb': 'ChromaDB',
        'loguru': 'Loguru',
        'yaml': 'PyYAML'
    }
    
    optional = {
        'openai': 'OpenAI API',
        'langchain': 'LangChain',
        'peft': 'PEFT (for LoRA)',
        'bitsandbytes': 'BitsAndBytes (for quantization)'
    }
    
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    for module, name in optional.items():
        try:
            __import__(module)
        except ImportError:
            warnings.append(name)
    
    if missing:
        return HealthCheckResult(
            component="Dependencies",
            status="error",
            message=f"Missing required: {', '.join(missing)}",
            details={"missing": missing, "optional_missing": warnings}
        )
    elif warnings:
        return HealthCheckResult(
            component="Dependencies",
            status="warning",
            message=f"Missing optional: {', '.join(warnings)}",
            details={"optional_missing": warnings}
        )
    else:
        return HealthCheckResult(
            component="Dependencies",
            status="ok",
            message="All dependencies installed"
        )


def check_gpu() -> HealthCheckResult:
    """Check GPU availability and memory."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_free = memory_total - memory_allocated
            
            return HealthCheckResult(
                component="GPU",
                status="ok",
                message=f"{device_name} ({memory_free:.1f}GB free)",
                details={
                    "device": device_name,
                    "total_gb": round(memory_total, 1),
                    "allocated_gb": round(memory_allocated, 1),
                    "free_gb": round(memory_free, 1)
                }
            )
        else:
            return HealthCheckResult(
                component="GPU",
                status="warning",
                message="No GPU available, using CPU",
                details={"cuda_available": False}
            )
    except Exception as e:
        return HealthCheckResult(
            component="GPU",
            status="error",
            message=f"Error checking GPU: {e}"
        )


def check_vector_db(vector_db_path: str) -> HealthCheckResult:
    """Check vector database health."""
    path = Path(vector_db_path)
    
    if not path.exists():
        return HealthCheckResult(
            component="Vector DB",
            status="error",
            message=f"Database not found at {path}"
        )
    
    try:
        from src.retrieval.vector_store import VectorStore, VectorStoreConfig
        
        config = VectorStoreConfig(persist_directory=str(path))
        store = VectorStore(config)
        stats = store.get_collection_stats()
        
        count = stats.get('count', 0)
        
        if count == 0:
            return HealthCheckResult(
                component="Vector DB",
                status="warning",
                message="Database is empty",
                details=stats
            )
        
        return HealthCheckResult(
            component="Vector DB",
            status="ok",
            message=f"{count:,} documents indexed",
            details=stats
        )
    except Exception as e:
        return HealthCheckResult(
            component="Vector DB",
            status="error",
            message=f"Error accessing database: {e}"
        )


def check_embedding_model(model_name: str = None) -> HealthCheckResult:
    """Check embedding model health."""
    if model_name is None:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        from src.embeddings.embedder import EmbeddingGenerator, EmbeddingConfig
        
        start = time.time()
        config = EmbeddingConfig(model_name=model_name)
        embedder = EmbeddingGenerator(config)
        
        # Test embedding
        test_text = "This is a test sentence."
        embedding = embedder.embed_text(test_text)
        
        load_time = time.time() - start
        
        return HealthCheckResult(
            component="Embedding Model",
            status="ok",
            message=f"Model loaded ({load_time:.1f}s, dim={embedder.dimension})",
            details={
                "model": model_name,
                "dimension": embedder.dimension,
                "load_time": round(load_time, 2)
            }
        )
    except Exception as e:
        return HealthCheckResult(
            component="Embedding Model",
            status="error",
            message=f"Error loading model: {e}"
        )


def check_data_directory(data_dir: str) -> HealthCheckResult:
    """Check data directory."""
    path = Path(data_dir)
    
    if not path.exists():
        return HealthCheckResult(
            component="Data Directory",
            status="error",
            message=f"Not found: {path}"
        )
    
    # Count files
    txt_files = list(path.glob("**/*.txt"))
    json_files = list(path.glob("**/*.json"))
    total_files = len(txt_files) + len(json_files)
    
    if total_files == 0:
        return HealthCheckResult(
            component="Data Directory",
            status="warning",
            message="No data files found",
            details={"path": str(path)}
        )
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    return HealthCheckResult(
        component="Data Directory",
        status="ok",
        message=f"{total_files} files ({size_mb:.1f}MB)",
        details={
            "txt_files": len(txt_files),
            "json_files": len(json_files),
            "total_size_mb": round(size_mb, 1)
        }
    )


def run_health_checks(vector_db_path: str = "./vector_db", data_dir: str = "../tunisian_heritage_data") -> List[HealthCheckResult]:
    """Run all health checks."""
    results = [
        check_dependencies(),
        check_gpu(),
        check_data_directory(data_dir),
        check_vector_db(vector_db_path),
        check_embedding_model()
    ]
    return results


def test_retrieval_quality(vector_db_path: str = "./vector_db", num_queries: int = 5) -> Dict[str, Any]:
    """Test retrieval quality with sample queries."""
    logger.info("Testing retrieval quality...")
    
    # Sample test queries
    test_queries = [
        "What is the history of Tunisia?",
        "Tunisian traditional food",
        "Revolution in Tunisia",
        "Ancient Carthage civilization",
        "Tunisian culture and traditions"
    ]
    
    results = {
        "queries": [],
        "avg_score": 0,
        "avg_time": 0
    }
    
    try:
        from src.pipeline.rag_pipeline import RAGPipeline, RAGConfig
        
        config = RAGConfig(vector_db_dir=vector_db_path)
        pipeline = RAGPipeline(config)
        pipeline.initialize()
        
        total_score = 0
        total_time = 0
        
        for query in test_queries[:num_queries]:
            start = time.time()
            response = pipeline.query(query, top_k=3)
            elapsed = time.time() - start
            
            query_result = {
                "query": query,
                "num_results": len(response.sources),
                "top_score": response.sources[0]["score"] if response.sources else 0,
                "time_seconds": round(elapsed, 2)
            }
            
            results["queries"].append(query_result)
            
            if response.sources:
                total_score += query_result["top_score"]
            total_time += elapsed
        
        results["avg_score"] = round(total_score / num_queries, 3) if num_queries > 0 else 0
        results["avg_time"] = round(total_time / num_queries, 2) if num_queries > 0 else 0
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


def profile_pipeline(vector_db_path: str = "./vector_db") -> Dict[str, Any]:
    """Profile pipeline performance."""
    logger.info("Profiling pipeline performance...")
    
    results = {
        "initialization": {},
        "embedding": {},
        "retrieval": {},
        "generation": {}
    }
    
    try:
        # Profile initialization
        start = time.time()
        from src.pipeline.rag_pipeline import RAGPipeline, RAGConfig
        config = RAGConfig(vector_db_dir=vector_db_path)
        pipeline = RAGPipeline(config)
        pipeline.initialize()
        results["initialization"]["time_seconds"] = round(time.time() - start, 2)
        
        # Profile embedding
        from src.embeddings.embedder import EmbeddingGenerator, EmbeddingConfig
        embedder = EmbeddingGenerator(EmbeddingConfig())
        
        test_texts = ["Test sentence " + str(i) for i in range(100)]
        
        start = time.time()
        embeddings = embedder.embed_texts(test_texts)
        embed_time = time.time() - start
        
        results["embedding"] = {
            "batch_size": 100,
            "total_time": round(embed_time, 2),
            "per_text_ms": round(embed_time * 1000 / 100, 2)
        }
        
        # Profile retrieval
        test_query = "What is Tunisian history?"
        
        start = time.time()
        response = pipeline.query(test_query)
        query_time = time.time() - start
        
        results["retrieval"] = {
            "query": test_query,
            "total_time": round(query_time, 2),
            "num_results": len(response.sources)
        }
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            results["memory_usage_mb"] = round(memory_mb, 1)
        except ImportError:
            pass
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


def view_logs(log_dir: str = "./logs", lines: int = 50) -> None:
    """View recent log entries."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_path}")
        return
    
    # Find most recent log file
    log_files = sorted(log_path.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if not log_files:
        print("No log files found")
        return
    
    latest = log_files[0]
    print(f"📄 Latest log: {latest}")
    print("=" * 60)
    
    with open(latest, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line, end='')


def export_diagnostics(output_path: str, vector_db_path: str = "./vector_db") -> None:
    """Export full diagnostics report to JSON."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "health_checks": [],
        "retrieval_test": {},
        "profiling": {}
    }
    
    # Health checks
    logger.info("Running health checks...")
    for result in run_health_checks(vector_db_path):
        report["health_checks"].append({
            "component": result.component,
            "status": result.status,
            "message": result.message,
            "details": result.details
        })
    
    # Retrieval test
    logger.info("Running retrieval tests...")
    report["retrieval_test"] = test_retrieval_quality(vector_db_path)
    
    # Profiling
    logger.info("Running profiling...")
    report["profiling"] = profile_pipeline(vector_db_path)
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Diagnostics report saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnostics and debugging tools for RAG pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # Health check
    health_parser = subparsers.add_parser("health", help="Run health checks")
    health_parser.add_argument("--vector-db", default="./vector_db")
    health_parser.add_argument("--data-dir", default="../tunisian_heritage_data")
    
    # Test retrieval
    retr_parser = subparsers.add_parser("test-retrieval", help="Test retrieval quality")
    retr_parser.add_argument("--vector-db", default="./vector_db")
    retr_parser.add_argument("--num-queries", type=int, default=5)
    
    # Profile
    prof_parser = subparsers.add_parser("profile", help="Profile performance")
    prof_parser.add_argument("--vector-db", default="./vector_db")
    
    # Logs
    logs_parser = subparsers.add_parser("logs", help="View recent logs")
    logs_parser.add_argument("--log-dir", default="./logs")
    logs_parser.add_argument("--lines", type=int, default=50)
    
    # Export
    export_parser = subparsers.add_parser("export", help="Export full diagnostics report")
    export_parser.add_argument("--output", default="./diagnostics_report.json")
    export_parser.add_argument("--vector-db", default="./vector_db")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("\n🔧 Tunisian Heritage RAG - Diagnostics")
    print("=" * 60)
    
    if args.command == "health":
        results = run_health_checks(args.vector_db, args.data_dir)
        print("\nHealth Check Results:")
        print("-" * 40)
        for result in results:
            print(result)
        
        # Summary
        errors = sum(1 for r in results if r.status == "error")
        warnings = sum(1 for r in results if r.status == "warning")
        
        print("-" * 40)
        if errors:
            print(f"❌ {errors} error(s) found")
        elif warnings:
            print(f"⚠️  {warnings} warning(s), but system operational")
        else:
            print("✅ All systems healthy!")
    
    elif args.command == "test-retrieval":
        results = test_retrieval_quality(args.vector_db, args.num_queries)
        print("\nRetrieval Test Results:")
        print("-" * 40)
        
        for q in results.get("queries", []):
            print(f"Query: {q['query'][:50]}...")
            print(f"  Results: {q['num_results']}, Top Score: {q['top_score']:.3f}, Time: {q['time_seconds']}s")
        
        print("-" * 40)
        print(f"Average Score: {results['avg_score']:.3f}")
        print(f"Average Time: {results['avg_time']:.2f}s")
    
    elif args.command == "profile":
        results = profile_pipeline(args.vector_db)
        print("\nProfiling Results:")
        print("-" * 40)
        print(json.dumps(results, indent=2))
    
    elif args.command == "logs":
        view_logs(args.log_dir, args.lines)
    
    elif args.command == "export":
        export_diagnostics(args.output, args.vector_db)
    
    else:
        print("Available commands: health, test-retrieval, profile, logs, export")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
