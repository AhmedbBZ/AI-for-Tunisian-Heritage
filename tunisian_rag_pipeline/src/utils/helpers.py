"""
Helper Utilities
================
Common utility functions used across the RAG pipeline.
"""

import os
import sys
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

from loguru import logger


# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(filepath: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha256, etc.)
        
    Returns:
        Hex digest of file hash
    """
    h = hashlib.new(algorithm)
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    
    return h.hexdigest()


def get_text_hash(text: str, algorithm: str = "md5") -> str:
    """
    Calculate hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm
        
    Returns:
        Hex digest
    """
    h = hashlib.new(algorithm)
    h.update(text.encode('utf-8'))
    return h.hexdigest()


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def list_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[Path]:
    """
    List files in a directory.
    
    Args:
        directory: Directory to search
        extensions: File extensions to include (e.g., ['.txt', '.json'])
        recursive: Search recursively
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for path in directory.glob(pattern):
        if path.is_file():
            if extensions is None or path.suffix.lower() in extensions:
                files.append(path)
    
    return sorted(files)


# =============================================================================
# TEXT UTILITIES
# =============================================================================

def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "..."
) -> str:
    """
    Truncate text to max length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def char_count(text: str, include_spaces: bool = True) -> int:
    """Count characters in text."""
    if include_spaces:
        return len(text)
    return len(text.replace(' ', ''))


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    import re
    
    # Handle multiple sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    import re
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


# =============================================================================
# TIMING UTILITIES
# =============================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Usage:
        @timer
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


@contextmanager
def timed_block(name: str = "Block"):
    """
    Context manager for timing code blocks.
    
    Usage:
        with timed_block("My operation"):
            ...
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} completed in {elapsed:.2f}s")


class Timer:
    """
    Simple timer class.
    
    Usage:
        timer = Timer()
        # ... do something ...
        print(f"Elapsed: {timer.elapsed:.2f}s")
    """
    
    def __init__(self):
        self.start_time = time.time()
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def reset(self) -> float:
        elapsed = self.elapsed
        self.start_time = time.time()
        return elapsed


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_process(
    items: List[Any],
    batch_size: int,
    process_func: Callable,
    show_progress: bool = True
) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        batch_size: Size of each batch
        process_func: Function to process each batch
        show_progress: Show progress bar
        
    Returns:
        Combined results
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        if show_progress:
            logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        batch_results = process_func(batch)
        
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results


def chunked(items: List[Any], size: int) -> List[List[Any]]:
    """
    Split list into chunks.
    
    Args:
        items: List to chunk
        size: Chunk size
        
    Returns:
        List of chunks
    """
    return [items[i:i + size] for i in range(0, len(items), size)]


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed, trying JSON")
    
    if config_path.suffix == '.json':
        return load_json(config_path)
    
    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB"
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        rotation: Log rotation setting
    """
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File handler
    if log_file:
        ensure_directory(Path(log_file).parent)
        logger.add(
            log_file,
            level=log_level,
            rotation=rotation,
            compression="zip"
        )


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_path_exists(path: Union[str, Path], path_type: str = "path") -> Path:
    """
    Validate that a path exists.
    
    Args:
        path: Path to validate
        path_type: Type of path (for error messages)
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If path doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path_type} not found: {path}")
    return path


def validate_file_exists(filepath: Union[str, Path]) -> Path:
    """Validate that a file exists."""
    path = validate_path_exists(filepath, "File")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    return path


def validate_directory_exists(dirpath: Union[str, Path]) -> Path:
    """Validate that a directory exists."""
    path = validate_path_exists(dirpath, "Directory")
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return path


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory stats in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024)
        }
    except ImportError:
        return {"error": "psutil not installed"}


def get_gpu_memory() -> Dict[str, Any]:
    """
    Get GPU memory usage.
    
    Returns:
        Dictionary with GPU memory stats
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "device": torch.cuda.get_device_name()
            }
        return {"available": False}
    except ImportError:
        return {"error": "torch not installed"}


# =============================================================================
# STRING FORMATTING
# =============================================================================

def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a simple text progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percent:.1%} ({current}/{total})"


if __name__ == "__main__":
    # Quick tests
    print("Helper Utilities")
    print("=" * 60)
    
    # Test timer
    timer = Timer()
    time.sleep(0.1)
    print(f"Timer test: {timer.elapsed:.3f}s")
    
    # Test text utilities
    text = "Hello world. This is a test. Another sentence!"
    sentences = split_into_sentences(text)
    print(f"Sentences: {sentences}")
    
    # Test formatting
    print(f"Format bytes: {format_bytes(1234567890)}")
    print(f"Format duration: {format_duration(3665)}")
    print(f"Progress bar: {create_progress_bar(45, 100)}")
